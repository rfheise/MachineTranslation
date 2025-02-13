from .Model import Model
import torch 
from torch import nn 
from .Transformer import PositionalEncoding
from ..Dataset.Language import get_language_loader 
from tqdm import tqdm
from ..Metrics.Acc import Acc 
from ..Metrics.Avg import Avg
import random 

device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    device = torch.device("mps")


class Transformer(nn.Module):

    def __init__(self, embed_dim, num_tokens):
        super().__init__()

        self.transformer = nn.Transformer(d_model = embed_dim,batch_first=True, nhead=2, num_encoder_layers=1, num_decoder_layers=1)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_tokens),
        )
        self.out_lang_embeddings = None
        self.in_lang_embeddings = None
        self.pos_enc = PositionalEncoding(embed_dim)
    
    def set_out_lang_embeddings(self, out_lang_embeddings):

        self.out_lang_embeddings = nn.Embedding.from_pretrained(out_lang_embeddings, freeze=False).to(device)
    
    def set_in_lang_embeddings(self, in_lang_embeddings):

        self.in_lang_embeddings = nn.Embedding.from_pretrained(in_lang_embeddings, freeze=False).to(device)

    def forward(self, X, Y, mask=None):
        X = self.in_lang_embeddings(X)
        Y = self.out_lang_embeddings(Y)
        X = self.pos_enc(X)
        Y = self.pos_enc(Y)
        out = self.transformer(X, Y, tgt_mask=mask, tgt_is_causal=True)
        out = self.linear(out)
        return out


class TransformerBoujee(Model):
    
    def __init__(self):
        super().__init__()
        self.transformer = None 
    
    def train(self, dataset, loss, batch_size=32):
        
        dataset.train_init()
        loader = get_language_loader(dataset.train, batch_size=batch_size, shuffle=True)
        self.lazy_init(dataset)
        self.transformer.train()
        l_avg = Avg("Loss")
        acc = Acc()
        for X,Y in tqdm(loader, total=len(loader)):
            
            X = X.to(device).long()
            Y = Y.to(device).long()
            mask = nn.Transformer.generate_square_subsequent_mask(Y.shape[1]).to(device)
            out = self.transformer(X,Y, mask).transpose(1,2)
            l = loss(out, Y)
            acc.compute_score(out.argmax(dim=1), Y)
            l_avg.compute_score(l)
            self.optim.zero_grad()
            l.backward()
            self.optim.step()

        ref = dataset.decode_sentence(X[1].cpu(), "inlang")
        out_sentence = dataset.decode_sentence(Y[1].cpu())
        sentence = dataset.decode_sentence(out.argmax(dim=1)[1].cpu())
        print("in:",ref,"\n")
        print("target:",out_sentence,"\n")
        print("out:",sentence,"\n")
        l_avg.display()
        acc.display()
    
    def lazy_init(self, dataset):
        if self.transformer is None:
            self.num_tokens = dataset.outlang.embeddings.shape[0]
            self.embed_dim = dataset.inlang.embeddings.shape[1]
            self.transformer = Transformer(self.embed_dim, self.num_tokens)
            self.transformer.set_in_lang_embeddings(dataset.inlang.embeddings)
            self.transformer.set_out_lang_embeddings(dataset.outlang.embeddings)
            self.transformer.to(device)
            self.optim = torch.optim.Adam(self.transformer.parameters())

    
    def pred_prob(self, dataset):

        pass

    def pred(self, dataset):

        pass

    def load(self, fname):

        pass 

    def init_model(self):

        pass

    def save(self, fname):
        pass