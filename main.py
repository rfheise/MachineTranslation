from .Train.train import train_model
from .Train.test import test_model
from .Model.TransformerBoujee import TransformerBoujee
from .Dataset.GerToEng import EngToGer
import torch
import torch.nn as nn
from .Log import Logger

def main():
    Logger.init_logger(wandb=False, print=True)
    model = TransformerBoujee()
    dataset = EngToGer()
    loss = nn.CrossEntropyLoss(ignore_index=0)
    epoch_start = 0
    epoch_end = 500
    metrics = []
    train_model(model, dataset, loss, epoch_start, epoch_end, "mk1.pth", metrics)
    # test_model(model, dataset,loss, "./translate/attempts/mk1.pth", metrics)

if __name__ == "__main__":
    main()