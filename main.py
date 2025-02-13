from .Train.train import train_model
from .Model.TransformerBoujee import TransformerBoujee
from .Dataset.GerToEng import EngToGer
import torch
import torch.nn as nn

def main():

    model = TransformerBoujee()
    dataset = EngToGer()
    loss = nn.CrossEntropyLoss(ignore_index=0)
    epoch_start = 0
    epoch_end = 500
    metrics = []
    train_model(model, dataset, loss, epoch_start, epoch_end, "bruh.pth", metrics)

if __name__ == "__main__":
    main()