from .Train.train import train_model
from .Train.test import test_model, infer
from .Model.TransformerBoujee import TransformerBoujee
from .Dataset.GerToEng import EngToGer
from .Dataset.EngToFr import EngToFr
import torch
import torch.nn as nn
from .Search.Search import greedy_search, beam_search
from .Log import Logger

def eng_to_ger():
    run_id = 'pdpfvxgj'
    Logger.init_logger(wandb=True, print=True, run_id=run_id)
    dataset = EngToGer()
    loss = nn.CrossEntropyLoss(ignore_index=0)
    lr_decay_step = 15
    lr = 2e-4
    lr_decay = .1
    epoch_start = 0
    epoch_end = lr_decay_step * 3
    batch_size = 256
    model = TransformerBoujee(lr, lr_decay, lr_decay_step, batch_size)
    metrics = []
    # fname = "./translate/attempts/ger_mk1/model-epoch-2.pth"
    fname = None
    train_model(model, dataset, loss, epoch_start, epoch_end, fname,"./translate/attempts/ger_mk1", metrics)
    search = beam_search
    # test_model(model, dataset,loss, "./translate/attempts/french.pth", search,metrics)
    # infer(model, dataset,"./translate/attempts/mk1.pth",search )

def eng_to_fr():
    run_id = 'tuplnc11'
    Logger.init_logger(wandb=True, print=True, run_id=run_id)
    
    dataset = EngToFr()
    loss = nn.CrossEntropyLoss(ignore_index=0)
    lr_decay_step = 3
    lr = 2e-4
    lr_decay = .1
    epoch_start = 0
    epoch_end = lr_decay_step * 3
    batch_size = 256
    model = TransformerBoujee(lr, lr_decay, lr_decay_step, batch_size)
    metrics = []
    # fname = "./translate/attempts/ger_mk1/model-epoch-2.pth"
    fname = None
    train_model(model, dataset, loss, epoch_start, epoch_end, fname,"./translate/attempts/fr_mk1", metrics)
    search = beam_search
    # test_model(model, dataset,loss, "./translate/attempts/french.pth", search,metrics)
    # infer(model, dataset,"./translate/attempts/mk1.pth",search )


if __name__ == "__main__":
    eng_to_ger()
    # eng_to_fr()
