from .Model import Model
import torch 
from torch import nn 
from .Transformer import PositionalEncoding
from ..Dataset.Language import get_language_loader 
from tqdm import tqdm
from ..Metrics.Acc import Acc 
from ..Metrics.Avg import Avg
from ..Metrics.Bleu import Bleu
from ..Search.Search import greedy_search
import random 
from ..Log import Logger

device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    device = "mps"


class Transformer(nn.Module):

    def __init__(self, embed_dim, num_tokens_in, num_tokens_out):
        super().__init__()

        self.transformer = nn.Transformer(d_model = embed_dim,batch_first=True, nhead=10, num_encoder_layers=6, num_decoder_layers=6)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_tokens_out),
        )
        self.out_lang_embeddings = nn.Embedding(num_tokens_out, embed_dim).to(device)
        self.in_lang_embeddings = nn.Embedding(num_tokens_in, embed_dim).to(device)
        self.pos_enc = PositionalEncoding(embed_dim).to(device)
    
    def set_out_lang_embeddings(self, out_lang_embeddings):

        self.out_lang_embeddings = nn.Embedding.from_pretrained(out_lang_embeddings, freeze=False).to(device)
    
    def set_in_lang_embeddings(self, in_lang_embeddings):

        self.in_lang_embeddings = nn.Embedding.from_pretrained(in_lang_embeddings, freeze=False).to(device)

    def forward(self, X, Y, mask=None):
        X = self.in_lang_embeddings(X)
        Y = self.out_lang_embeddings(Y)
        X = self.pos_enc(X)
        Y = self.pos_enc(Y)
        if mask is not None:
            out = self.transformer(X, Y, tgt_mask=mask, tgt_is_causal=True)
        else:
            out = self.transformer(X, Y)
        out = self.linear(out)
        return out


class TransformerBoujee(Model):
    
    def __init__(self):
        super().__init__()
        self.transformer = None 
    
    def train(self, dataset, loss, epoch = 0, batch_size=32):
        
        dataset.train_init()
        loader = get_language_loader(dataset.train, batch_size=batch_size, shuffle=True)
        self.lazy_init(dataset)
        self.transformer.train()
        l_avg = Avg("Loss")
        acc = Acc()
        for X,Y in tqdm(loader, total=len(loader)):
            
            X = X.to(device).long()
            Y = Y.to(device).long()
            decoder_input = Y[:,:-1]
            target = Y[:,1:]
            mask = nn.Transformer.generate_square_subsequent_mask(decoder_input.shape[1]).to(device)
            out = self.transformer(X,decoder_input, mask).transpose(1,2)
            l = loss(out, target)
            acc.compute_score(out.detach().argmax(dim=1), target.detach())
            l_avg.compute_score(l.detach())
            self.optim.zero_grad()
            l.backward()
            self.optim.step()

        ref = dataset.decode_sentence(X[1].detach().cpu(), "inlang")
        out_sentence = dataset.decode_sentence(Y[1].detach().cpu())
        sentence = dataset.decode_sentence(out.argmax(dim=1)[1].detach().cpu())
        # print("in:",ref,"\n")
        # print("target:",out_sentence,"\n")
        # print("out:",sentence,"\n")
        # l_avg.display()
        # acc.display()
        Logger.log({"loss":l_avg.ret_avg(), "acc":acc.ret_avg(), "in":ref,"target":out_sentence,"out":sentence, "epoch":epoch})
    
    def lazy_init(self, dataset):
        if self.transformer is None:
            self.num_tokens_out = dataset.outlang.embeddings.shape[0]
            self.num_tokens_in = dataset.inlang.embeddings.shape[0]
            self.embed_dim = dataset.inlang.embeddings.shape[1]
            self.transformer = Transformer(self.embed_dim, self.num_tokens_in, self.num_tokens_out)
            self.transformer.set_in_lang_embeddings(dataset.inlang.embeddings)
            self.transformer.set_out_lang_embeddings(dataset.outlang.embeddings)
            self.transformer = self.transformer.to(device)
            self.optim = torch.optim.Adam(self.transformer.parameters(), lr=1e-4)

    def test(self, dataset, loss, metrics=[]):
    
        dataset.val_init()
        loader = get_language_loader(dataset.val, batch_size=32, shuffle=True)
        self.lazy_init(dataset)
        self.transformer.eval()
        l_avg = Avg("Loss")
        acc = Acc()
        bleu = Bleu()

        with torch.no_grad():
            for X,Y in tqdm(loader, total=len(loader)):
                
                X = X.to(device).long()
                Y = Y.to(device).long()
                out = greedy_search(self, X, dataset.outlang)
                # print(out.shape)
                # l = loss(out, Y)
                # acc.compute_score(out.detach().argmax(dim=1), Y.detach())
                # l_avg.compute_score(l.detach())
                
                # bleu.compute_score(dataset.decode_sentences(out.detach()), dataset.decode_sentences(Y.detach()))

                ref = dataset.decode_sentence(X[1].detach().cpu(), "inlang")
                out_sentence = dataset.decode_sentence(Y[1].detach().cpu())
                sentence = dataset.decode_sentence(out[1].detach().cpu())
                print("in:",ref,"\n")
                print("target:",out_sentence,"\n")
                print("out:",sentence,"\n")
        # l_avg.display()
        # acc.display()
        Logger.log({"loss":l_avg.ret_avg(), "acc":acc.ret_avg(),"bleu":bleu.ret_avg(),
                     "in":ref,"target":out_sentence,"out":sentence})
    def pred_prob(self, X, Y):
        # mask = nn.Transformer.generate_square_subsequent_mask(Y.shape[1]).to(device)
        # print(mask)
        # print(self.transformer(X, Y, mask).shape)
        # exit()
        return nn.Softmax(dim = 2)(self.transformer(X, Y))

    def pred(self, dataset):

        pass

    def load(self, fname):

        state_dicts = torch.load(fname,map_location=device)
        self.num_tokens_in = state_dicts["num_tokens_in"]
        self.num_tokens_out = state_dicts["num_tokens_out"]
        self.embed_dim = state_dicts["embed_dim"]
        self.transformer = Transformer(self.embed_dim, self.num_tokens_in, self.num_tokens_out)
        self.transformer.load_state_dict(state_dicts["transformer"])
        self.transformer.in_lang_embeddings.load_state_dict(state_dicts["inlang_embed"])
        self.transformer.out_lang_embeddings.load_state_dict(state_dicts["outlang_embed"])
        self.transformer = self.transformer.to(device)
        self.optim = torch.optim.Adam(self.transformer.parameters())
        self.optim.load_state_dict(state_dicts["optim"])
        return state_dicts["epoch"]
    
    def init_model(self):

        pass

    def save(self, fname, epoch=0):
        torch.save({
            "transformer":self.transformer.state_dict(),
            "inlang_embed":self.transformer.in_lang_embeddings.state_dict(),
            "outlang_embed": self.transformer.out_lang_embeddings.state_dict(),
            "optim":self.optim.state_dict(),
            "num_tokens_out": self.num_tokens_out,
            "num_tokens_in": self.num_tokens_in,
            "epoch":epoch,
            "embed_dim":self.embed_dim,
        },fname)
