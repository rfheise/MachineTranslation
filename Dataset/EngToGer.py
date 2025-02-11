from Dataset import Dataset, Data

class EngToGer(Dataset):


    def __init__(self, fname):
        super().__init__(fname) 

    def load_file(self, fname):
        
        
