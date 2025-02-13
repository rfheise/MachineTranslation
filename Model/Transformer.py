import torch 
from torch import nn
from .Model import Model
import math
from ..Dataset.Language import get_language_loader
from tqdm import tqdm
from ..Metrics.Avg import Avg
from ..Metrics.Bleu import Bleu
import os
from ..Metrics.Acc import Acc

device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    device = torch.device("mps")

class SequentialMultiArg(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if isinstance(inputs, tuple):
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a matrix of [max_len, d_model] representing positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices
        
        # Instead of transposing, just unsqueeze to have shape [1, max_len, d_model]
        pe = pe.unsqueeze(0)  
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        """
        # x.size(1) is the sequence length, so self.pe[:, :x.size(1)] has shape [1, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    
class EncoderBlock(nn.Module):

    def __init__(self, embed_dim=300, heads=5):
        super().__init__()
        self.head = nn.MultiheadAttention(embed_dim, heads, dropout=0.1, batch_first=True)
        self.drop1 = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.linear = SequentialMultiArg(
            nn.Linear(embed_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, embed_dim),
            nn.Dropout(0.1)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, X):
        X_prime = self.drop1(self.head(X, X, X)[0])
        X = self.norm1(X + X_prime)
        X_prime = self.linear(X)
        X = self.norm2(X + X_prime)
        return X


class DecoderBlock(nn.Module):

    def __init__(self, embed_dim=300, heads=5):
        super().__init__()
        self.head1 = nn.MultiheadAttention(embed_dim, heads, dropout=0.1, batch_first=True)
        self.drop1 = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.head2 = nn.MultiheadAttention(embed_dim, heads, dropout=0.1, batch_first=True)
        self.drop2 = nn.Dropout(0.1)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffnn = SequentialMultiArg(
            nn.Linear(embed_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, embed_dim),
            nn.Dropout(0.1)
        )
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, X, enc_out, mask=False): 
        if mask:
            attn_mask = nn.Transformer.generate_square_subsequent_mask(X.shape[1]).to(device)
        else:
            attn_mask = None
        X_prime = self.drop1(self.head1(X, X, X,attn_mask=attn_mask)[0])
        X = self.norm1(X + X_prime)
        X_prime = self.drop2(self.head2(X, enc_out, enc_out)[0])
        X = self.norm2(X + X_prime)
        X_prime = self.ffnn(X)
        X = self.norm3(X + X_prime)
        return X

        

class TransformerEncoder(nn.Module):
    
    def __init__(self, embed_dim=300, heads=10, num_layers=3,):
        super().__init__()
        self.encoder = SequentialMultiArg(
            *[EncoderBlock(embed_dim, heads) for _ in range(num_layers)]
        )
        self.pos_enc = PositionalEncoding(embed_dim)
        self.in_lang_embeddings = None 

    def set_in_lang_embeddings(self, in_lang_embeddings):
        self.in_lang_embeddings = nn.Embedding.from_pretrained(in_lang_embeddings, freeze=False).to(device)
    
    def forward(self, X):
        if self.in_lang_embeddings is None:
            raise ValueError("in_lang_embeddings not set")
        X = X.long()
        X = self.in_lang_embeddings(X)
        X = self.pos_enc(X)
        X = self.encoder(X)
        return X
    
class TransformerDecoder(nn.Module):

    def __init__(self, num_tokens, embed_dim=300, heads=5, num_layers=3):
        super().__init__()
        self.decoder_blocks = [DecoderBlock(embed_dim, heads).to(device) for _ in range(num_layers)]
        self.out = SequentialMultiArg(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_tokens),
        )
        self.pos_enc = PositionalEncoding(embed_dim)
        self.out_lang_embeddings = None 
    
    def set_out_lang_embeddings(self, out_lang_embeddings):
        self.out_lang_embeddings = nn.Embedding.from_pretrained(out_lang_embeddings, freeze=False).to(device)
    
    def generate_square_subsequent_mask(self,sz):
        """
        Creates a square mask for the sequence.
        The masked positions are filled with -inf. Unmasked positions are 0.
        This is used to prevent the decoder from "seeing" future tokens.
        """
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def forward(self, Y, enc_out):
        if self.out_lang_embeddings is None:
            raise ValueError("out_lang_embeddings not set")
        Y = Y.long()
        Y = self.out_lang_embeddings(Y)
        Y = self.pos_enc(Y)
        if self.training:
            mask = True
        else:
            mask = False
        for i, block in enumerate(self.decoder_blocks):
            Y = block(Y, enc_out,(mask and i == 0))
        return self.out(Y)

class Transformer(Model):
    
    def __init__(self, embed_dim=300, heads=5, num_layers=3):
        self.embed_dim = embed_dim
        self.heads = heads 
        self.num_layers = num_layers
        self.encoder = None 
        self.decoder = None
    
    def train(self, dataset, loss, batch_size=32):
        if self.encoder is None:
            self.lazy_init(dataset, "bruh.pth")
        dataset.train_load()
        loader = get_language_loader(dataset.train, batch_size=batch_size)
        l_avg = Avg("Loss")
        acc = Acc()
        for X,Y in tqdm(loader, total=len(loader)):
            
            X = X.to(device).long()
            Y = Y.to(device).long()
            enc_out = self.encoder(X)
            out = self.decoder(Y, enc_out).transpose(1, 2)
            l = loss(out, Y)
            acc.compute_score(out.argmax(dim=1), Y)
            l_avg.compute_score(l)
            self.optim.zero_grad()
            l.backward()
            self.optim.step()
            sentence = dataset.decode_sentence(out.argmax(dim=1)[0])
        print(sentence)
        l_avg.display()
        acc.display()
        self.save("bruh.pth")

    def lazy_init(self, dataset, fname=None):
        self.num_tokens = dataset.outlang.embeddings.shape[0]
        self.encoder = TransformerEncoder(self.embed_dim, self.heads, self.num_layers).to(device)
        self.decoder = TransformerDecoder(self.num_tokens, self.embed_dim, self.heads, self.num_layers).to(device)
        # if fname is None or not os.path.exists(fname):
        self.encoder.set_in_lang_embeddings(dataset.inlang.embeddings)
        self.decoder.set_out_lang_embeddings(dataset.outlang.embeddings)
        self.optim = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=1e-4)
        if fname is not None and os.path.exists(fname):
            self.load(fname)     

    def pred_prob(self, dataset):
        if self.decoder is None:
            self.lazy_init(dataset)
        pass
        
    def pred(self, dataset):
        pass

    def load(self, fname):
        state_dicts = torch.load(fname,weights_only=False)
        self.encoder.load_state_dict(state_dicts["encoder"])
        self.decoder.load_state_dict(state_dicts["decoder"])
        self.optim.load_state_dict(state_dicts["optim"])
    
    
    def save(self, fname):

        torch.save({"encoder":self.encoder.state_dict(),
                    "decoder":self.decoder.state_dict(),
                    "optim": self.optim.state_dict()
                    }, fname)