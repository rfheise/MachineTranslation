from ..Log import Logger as log
import os

def train_model(model, dataset, loss, epoch_start, epoch_end,fname="bruh.pth", metrics=[]):
    
    if os.path.exists(fname):
        epoch_start = model.load(fname)
    
    for epoch in range(epoch_start, epoch_end + 1):
        
        print(f"--------- Epoch {epoch} ---------")

        model.train(dataset, loss, epoch)
        model.save(fname, epoch)
        