from ..Log import Logger as log


def train_model(model, dataset, loss, epoch_start, epoch_end):

    for epoch in range(epoch_start, epoch_end + 1):
        
        log.msg(f"--------- Epoch {epoch} ---------")

        model.train(dataset, loss)
        
        