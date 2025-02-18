from .Train.train import train_model
from .Train.test import test_model, infer
from .Model.TransformerBoujee import TransformerBoujee
from .Dataset.GerToEng import EngToGer
from .Dataset.EngToFr import EngToFr
import torch
import torch.nn as nn
from .Search.Search import greedy_search, beam_search
from .Log import Logger

def main():
    Logger.init_logger(wandb=False, print=True)
    model = TransformerBoujee()
    dataset = EngToFr()
    loss = nn.CrossEntropyLoss(ignore_index=0)
    epoch_start = 0
    epoch_end = 500
    metrics = []
    # train_model(model, dataset, loss, epoch_start, epoch_end, "mk1.pth", metrics)
    search = beam_search
    test_model(model, dataset,loss, "./translate/attempts/french.pth", search,metrics)
    # infer(model, dataset,"./translate/attempts/mk1.pth",search )

if __name__ == "__main__":
    main()
