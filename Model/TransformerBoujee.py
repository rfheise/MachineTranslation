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
import collections

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
        self.use_scaler = torch.cuda.is_available()
        if self.use_scaler:
            torch.backends.cudnn.benchmark = True
        else:
            self.scaler = None
    
    def train(self, dataset, loss, epoch = 0, batch_size=32):
        
        dataset.train_init()
        loader = get_language_loader(dataset.train, batch_size=batch_size, shuffle=True)
        self.lazy_init(dataset)
        self.transformer.train()
        l_avg = Avg("Loss")
        acc = Acc()
        masks = []
        for i in range(150):
            masks.append(nn.Transformer.generate_square_subsequent_mask(i).to(device))
        for X,Y in tqdm(loader, total=len(loader)):
            X = X.to(device)
            Y = Y.to(device)
            decoder_input = Y[:,:-1]
            target = Y[:,1:]
            # mask = nn.Transformer.generate_square_subsequent_mask(decoder_input.shape[1]).to(device)
            mask = masks[decoder_input.shape[1]]
            self.optim.zero_grad()
            if self.scaler is None:
                out = self.transformer(X,decoder_input, mask).transpose(1,2)
                l = loss(out, target)
                l.backward()
                self.optim.step()
            else:
                with torch.amp.autocast(device_type=device):
                    out = self.transformer(X,decoder_input, mask).transpose(1,2)
                    l = loss(out, target)
                self.scaler.scale(l).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
            acc.compute_score(out.detach().argmax(dim=1), target.detach())
            l_avg.compute_score(l.detach())
            

        ref = dataset.decode_sentence(X[1].detach().cpu(), "inlang")
        out_sentence = dataset.decode_sentence(Y[1].detach().cpu())
        sentence = dataset.decode_sentence(out.argmax(dim=1)[1].detach().cpu())
        print("in:",ref,"\n")
        print("target:",out_sentence,"\n")
        print("out:",sentence,"\n")
        # l_avg.display()
        # acc.display()
        l_test, bleu_test, acc_test = self.test(dataset, loss)
        Logger.log({"loss_train":l_avg.ret_avg(), "acc_train":acc.ret_avg(),
                    "loss_val": l_test.ret_avg(), "bleu_val": bleu_test.ret_avg(),
                    "acc_val": acc_test.ret_avg(),
                     "epoch":epoch})
    
    def lazy_init(self, dataset):
        if self.transformer is None:
            # torch.set_float32_matmul_precision("high")
            self.num_tokens_out = dataset.outlang.embeddings.shape[0]
            self.num_tokens_in = dataset.inlang.embeddings.shape[0]
            self.embed_dim = dataset.inlang.embeddings.shape[1]
            self.transformer = Transformer(self.embed_dim, self.num_tokens_in, self.num_tokens_out)
            self.transformer.set_in_lang_embeddings(dataset.inlang.embeddings)
            self.transformer.set_out_lang_embeddings(dataset.outlang.embeddings)
            # self.transformer = torch.compile(self.transformer)
            self.transformer = self.transformer.to(device)
            self.optim = torch.optim.Adam(self.transformer.parameters(), lr=1e-4)
            if self.use_scaler:
                self.scaler = torch.amp.GradScaler()
            else:
                self.scaler = None
            

    def test(self, dataset, loss, search=None, metrics=[]):
    
        dataset.val_init()
        loader = get_language_loader(dataset.val, batch_size=2, shuffle=True)
        self.lazy_init(dataset)
        self.transformer.eval()
        l_avg = Avg("Loss")
        acc = Acc()
        bleu = Bleu()
        with torch.no_grad():
            for X,Y in tqdm(loader, total=len(loader)):
                
                X = X.to(device)
                Y = Y.to(device)
                decoder_input = Y[:,:-1]
                target = Y[:,1:]
                if search is None:
                    mask = nn.Transformer.generate_square_subsequent_mask(decoder_input.shape[1]).to(device)
                    out = self.transformer(X, decoder_input.to(device), mask).transpose(1,2)
                    l = loss(out, target)
                    acc.compute_score(out.detach().argmax(dim=1), target.detach())
                    l_avg.compute_score(l.detach())
                    out = out.argmax(dim=1)
                else:
                    out = search(self, X, dataset.outlang)
                t_prob = self.compute_p_sentence(X,Y, dataset.outlang.get_token("<EOS>"))
                out_prob = self.compute_p_sentence(X,out,dataset.outlang.get_token("<EOS>"))
                bleu.compute_score(dataset.decode_sentences(out.detach()), dataset.decode_sentences(target.detach()))
                # ref = dataset.decode_sentence(X[0].detach().cpu(), "inlang")
                # out_sentence = dataset.decode_sentence(Y[0].detach().cpu())
                # sentence = dataset.decode_sentence(out[0].detach().cpu())
                # print("in:",ref,"\n")
                # print("target:",out_sentence,t_prob[0],"\n")
                # print("out:",sentence,out_prob[0],"\n")
        return l_avg, bleu, acc
    
    def compute_p_sentence(self, X, Y, eos_token):

        decoder_input = Y[:,:-1]
        target = Y[:,1:]
        probs = torch.log(self.pred_prob(X, decoder_input))
        vals = torch.gather(probs,2, index=target.unsqueeze(-1)).reshape(decoder_input.shape)
        # print(target)
        # print(vals)
        #remove all values that occur after EOS 
        batch, seq_len = target.shape

        # Create an index tensor for the sequence positions:
        idx = torch.arange(seq_len, device=target.device).unsqueeze(0).expand(batch, seq_len)
        
        # Create a mask where each position is True if tensor equals eos_token:
        eos_mask = (target == eos_token)
        
        # For each row, replace positions where there's no eos with a large value (seq_len)
        # so that when we take min, we get the index of the first eos.
        masked_idx = torch.where(eos_mask, idx, torch.full_like(idx, seq_len))
        first_eos_idx = masked_idx.min(dim=1)[0]  # shape: (batch,)
        
        # Build a mask that is True for positions <= first_eos index for each row.
        keep_mask = idx <= first_eos_idx.unsqueeze(1)
    
        vals = vals * keep_mask
        return vals.sum(dim=1, keepdim=True)

        
    def pred_prob(self, X, Y):
        mask = nn.Transformer.generate_square_subsequent_mask(Y.shape[1]).to(device)
        # print(mask)
        # print(self.transformer(X, Y, mask).shape)
        # exit()
        return nn.Softmax(dim = 2)(self.transformer(X, Y,mask))

    def pred(self, dataset):

        pass

    def load(self, fname):

        state_dicts = torch.load(fname,map_location=device)
        self.num_tokens_in = state_dicts["num_tokens_in"]
        self.num_tokens_out = state_dicts["num_tokens_out"]
        self.embed_dim = state_dicts["embed_dim"]
        self.transformer = Transformer(self.embed_dim, self.num_tokens_in, self.num_tokens_out)
        # self.transformer = torch.compile(self.transformer)
        self.transformer.load_state_dict(state_dicts["transformer"])
        self.transformer = self.transformer.to(device)
        self.transformer.in_lang_embeddings = self.transformer.in_lang_embeddings.to(device)
        self.transformer.out_lang_embeddings = self.transformer.out_lang_embeddings.to(device)
        self.transformer.in_lang_embeddings.load_state_dict(state_dicts["inlang_embed"])
        self.transformer.out_lang_embeddings.load_state_dict(state_dicts["outlang_embed"])
        self.optim = torch.optim.Adam(self.transformer.parameters(), lr = 1e-5)
        self.optim.load_state_dict(state_dicts["optim"])
        if self.use_scaler:
            self.scaler = torch.amp.GradScaler()
            self.scaler.load_state_dict(state_dicts["scaler"])
        
        return state_dicts["epoch"]
    
    def init_model(self):

        pass

    def save(self, fname, epoch=0):
        args = {}
        if self.scaler:
            args["scaler"] = self.scaler.state_dict()
        torch.save({
            "transformer":self.transformer.state_dict(),
            "inlang_embed":self.transformer.in_lang_embeddings.state_dict(),
            "outlang_embed": self.transformer.out_lang_embeddings.state_dict(),
            "optim":self.optim.state_dict(),
            "num_tokens_out": self.num_tokens_out,
            "num_tokens_in": self.num_tokens_in,
            "epoch":epoch,
            "embed_dim":self.embed_dim,
            **args
        },fname)

    @staticmethod
    def remove_prefix_from_state_dict(state_dict, prefix="_orig_mod."):
        """Remove a specific prefix from keys in the state dictionary."""
        new_state_dict = {}
        for key, value in state_dict.items():
            # print(key, type(key), type(value))
            if type(value) == dict or type(value) == collections.OrderedDict:
                value = TransformerBoujee.remove_prefix_from_state_dict(value)
            if type(key) == str and key.startswith(prefix):
                new_state_dict[key[len(prefix):]] = value
            else:
                new_state_dict[key] = value
        return new_state_dict
