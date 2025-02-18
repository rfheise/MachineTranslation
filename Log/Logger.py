
import wandb


class Logger():
    logs = []
    def __init__(self):
        pass

    def init_logger(wandb=True, print=True, run_id=None):
        if wandb:
            Logger.logs.append(WandB(run_id))
        if print:
            Logger.logs.append(Printer())

    def log(log_dict):

        for log in Logger.logs:
            log.msg(**log_dict)

class Log():
    def __init__(self, *args):
        pass

    def msg(*args, **kwargs):
        pass 

class WandB(Log):


    def __init__(self, run_id):
        self.run_id = run_id
        wandb.init(project = "Rosetta",id=run_id,resume="allow")
    
    def msg(self, *args,**kwargs):
        if "epoch" in kwargs:
            wandb.log(kwargs,step=kwargs['epoch'])
        else:
            wandb.log(kwargs)

class Printer(Log):

    def __init__(self, *args):
        pass 

    def msg(self, *args,**kwargs):

        for key in kwargs:
            print(f"{key}:{kwargs[key]}")