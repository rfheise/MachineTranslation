from .Train.train import train_model
from .Train.test import test_model, infer
from .Model.TransformerBoujee import TransformerBoujee
from .Dataset.GerToEng import EngToGer
from .Dataset.EngToFr import EngToFr
import torch
import torch.nn as nn
from .Search.Search import greedy_search, beam_search
from .Log import Logger
from .Dataset.Language import Embeddings

def eng_to_ger():
    run_id = 'ww8nd9id'
    Logger.init_logger(wandb=True, print=True, run_id=run_id)
    dataset = EngToGer(byte_pair=True)
    print(Embeddings.special_toks_val["<SOS>"])
    loss = nn.CrossEntropyLoss(ignore_index=Embeddings.special_toks_val["<PAD>"])
    lr_decay_step = 15
    batch_size = 128
    lr = 2e-4 * (256 / batch_size)
    lr_decay = .1
    epoch_start = 0
    # epoch_end = lr_decay_step * 3
    epoch_end = 50
    model = TransformerBoujee(lr, lr_decay, lr_decay_step, batch_size, epoch_end - epoch_start + 1)
    metrics = []
    # fname = "./translate/attempts/ger_mk1/model-epoch-5.pth"
    fname = None
    train_model(model, dataset, loss, epoch_start, epoch_end, fname,"./translate/attempts/ger_mk2", metrics)
    search = beam_search
    # test_model(model, dataset,loss, "./translate/attempts/french.pth", search,metrics)
    # infer(model, dataset,"./translate/attempts/mk1.pth",search )

def eng_to_fr():
    # run_id = 'tuplnc11'
    run_id = 'vrsw5bvh'
    run_id = 'z17rkobl'
    Logger.init_logger(wandb=False, print=True, run_id=run_id)
    
    dataset = EngToFr()
    print(Embeddings.special_toks_val["<PAD>"])
    loss = nn.CrossEntropyLoss(ignore_index=Embeddings.special_toks_val["<PAD>"])
    lr_decay_step = 7
    lr = 2e-4
    lr_decay = .1
    epoch_start = 0
    epoch_end = lr_decay_step * 3
    batch_size = 256
    model = TransformerBoujee(lr, lr_decay, lr_decay_step, batch_size)
    metrics = []
    for i in range(12):
        print("epoch " + str(i))
        fname = f"./translate/attempts/fr_mk2/model-epoch-{i}.pth"
    #fname = None
    #train_model(model, dataset, loss, epoch_start, epoch_end, fname,"./translate/attempts/fr_mk2", metrics)
        search = beam_search
        test_model(model, dataset,loss, fname, search,metrics)
    #infer(model, dataset,fname,search )


if __name__ == "__main__":
    eng_to_ger()
    # eng_to_fr()
