from ..Log import Logger as log
import os

def train_model(model, dataset, loss, epoch_start, epoch_end,fname="bruh.pth", metrics=[]):

    if os.path.exists(fname):
        model.load(fname)
    
    for epoch in range(epoch_start, epoch_end + 1):
        
        log.msg(f"--------- Epoch {epoch} ---------")

        model.train(dataset, loss)
        model.save(fname)
        