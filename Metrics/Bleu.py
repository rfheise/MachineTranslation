from .Metric import Metric
from torcheval.metrics.functional import bleu_score

class Bleu(Metric):
    
    def __init__(self, store=True):
        super().__init__("Bleu",store)
        # self.bleu_metric = BLEUScore(n_gram=4, smooth=True)

    def compute_score_child(self, *args):
        output, target = args 
        try:
            val = bleu_score(output, target, n_gram=4)
            return val.item(), 1
        except:
            return 0, 1



