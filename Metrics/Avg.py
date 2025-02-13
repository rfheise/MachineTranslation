from .Metric import Metric 

class Avg(Metric):

    def __init__(self,name, store=True):
        super().__init__(name, store)

    
