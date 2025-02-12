
from torch import nn
import torch.nn.functional as F
import torch
from .Model import Model

class Attention(nn.Module):

    def __init__(self, d_k, d_v, d_x,d_m):
        super().__init__()
        self.d_x = d_x
        self.d_v = d_v 
        self.d_k = d_k
        self.init_params()

    def init_params(self):

        self.W_k = nn.Parameter(torch.empty(self.d_k, self.d_x))
        nn.init.xavier_uniform_(self.W_k)
        self.W_q = nn.Parameter(torch.empty(self.d_k, self.d_x))
        nn.init.xavier_uniform_(self.W_q)
        self.W_v = nn.Parameter(torch.empty(self.d_v, self.d_x))
        nn.init.xavier_uniform_(self.W_v)

    def forward(self,Q,K,V):
                

        Q_prime = torch.bmm(Q, self.W_q)
        K_prime = torch.bmm(K, self.W_k)
        V_prime = torch.bmm(V, self.W_v)
        
        z = torch.mm(Q_prime, K_prime.T)
        scores = F.softmax(z/self.d_k**.5)

        return scores @ V_prime
    
class MultiHeadAttention(nn.Module):

    def __init__(self, heads, d_k, d_v, d_x, d_m):
        super().__init__()
        self.heads = heads
        self.d_x = d_x
        self.d_v = d_v 
        self.d_k = d_k
        self.d_m = d_m
        self.init_params()
    
    def init_params(self):

        self.W_k = nn.Parameter(torch.empty(self.d_k, self.heads, self.d_x))
        nn.init.xavier_uniform_(self.W_k)
        self.W_q = nn.Parameter(torch.empty(self.d_k, self.heads, self.d_x))
        nn.init.xavier_uniform_(self.W_q)
        self.W_v = nn.Parameter(torch.empty(self.d_v, self.heads, self.d_x))
        nn.init.xavier_uniform_(self.W_v)
        self.W_o = nn.Parameter(torch.empty(self.d_m, self.heads* self.d_v))
        nn.init.xavier_uniform_(self.W_o)

    def forward(self,Q,K,V):
                

        Q_prime = torch.bmm(Q, self.W_q)
        K_prime = torch.bmm(K, self.W_k)
        V_prime = torch.bmm(V, self.W_v)
        
        z = torch.mm(Q_prime, K_prime.T)
        scores = F.softmax(z/self.d_k**.5)
        heads = scores @ V_prime 

        return torch.bmm(self.W_o, heads)
    
class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.block =  nn.Sequential(
            MultiHeadAttention(),
            AddNorm(),
            FFNN(),
            AddNorm(),
        )

    def forward(self, X):
        return self.block(X,X,X)

class Transformer(nn.Module):

    def __init__(self, n):
        super().__init__()
        encoder = nn.Sequential(Embedding(), 
                                PositionalEncoding(),
                                *[EncoderBlock() for i in range(n)])
        decoder = nn.Sequential(
            
        )

    def forward(self):
        pass

