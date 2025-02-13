from .Metric import Metric 
import torch

class Acc(Metric):


    def __init__(self):
        super().__init__("accuracy")
    
    def compute_score_child(self, *args):
        
        out, target = args 
        out = torch.min((out == target).int(), dim=1)

        return out.values.sum(), len(out)