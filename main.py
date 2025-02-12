from .Train.train import train_model
from .Model.Transformer import Transformer
from .Dataset.GerToEng import EngToGer
import torch
import torch.nn as nn

def main():

    model = Transformer()
    dataset = EngToGer()
    loss = nn.CrossEntropyLoss(ignore_index=0)
    epoch_start = 0
    epoch_end = 10
    metrics = []
    train_model(model, dataset, loss, epoch_start, epoch_end, metrics)

if __name__ == "__main__":
    main()