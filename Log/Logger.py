
import wandb


class Logger():
    logs = []
    def __init__(self):
        pass

    def init_logger(wandb=True, print=True):
        if wandb:
            Logger.logs.append(WandB())
        if print:
            Logger.logs.append(Printer())

    def log(log_dict):

        for log in Logger.logs:
            log.msg(**log_dict)

class Log():
    def __init__(self):
        pass

    def msg(*args, **kwargs):
        pass 

class WandB(Log):


    def __init__(self):
        wandb.init(project = "Rosetta")
    
    def msg(*args,**kwargs):
        if "epoch" in kwargs:
            wandb.log(kwargs,step=kwargs['epoch'])
        else:
            wandb.log(kwargs)

class Printer(Log):

    def __init__(self):
        pass 

    def msg(*args,**kwargs):

        for key in kwargs:
            print(f"{key}:{kwargs[key]}")