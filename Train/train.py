from ..Log import Logger as log
import os

def train_model(model, dataset, loss, epoch_start, epoch_end,fname,dirname, metrics=[]):
    
    if fname is not None and os.path.exists(fname):
        epoch_start = model.load(fname)
        epoch_start += 1
    
    for epoch in range(epoch_start, epoch_end + 1):
        
        print(f"--------- Epoch {epoch} ---------")

        model.train(dataset, loss, epoch)
        save_fname = os.path.join(dirname,f"model-epoch-{epoch}.pth")
        model.save(save_fname, epoch)
        