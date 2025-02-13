

class Metric():

    def __init__(self, name, store=True):

        self.val = 0
        self.tot = 0 
        self.store = store
        self.name = name

    def compute_score(self, *args):
        
        if self.store:
            val,tot = self.compute_score_child(*args)
            self.add_to_tot(val, tot)
        return val, tot


    def compute_score_child(self, *args):
        
        l = args[0]
        return l, 1

    def add_to_tot(self, val, tot = 1):

        self.val += val 
        self.tot += tot 
    
    def ret_avg(self):
        if self.tot == 0:
            return 0
        return self.val/self.tot

    def reset(self):
        self.val = 0
        self.tot = 0
    
    def display(self):
        print(f"{self.name}:{self.ret_avg()}")