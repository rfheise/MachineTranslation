from torch.utils.data import Dataset as DS

class Dataset():

    def __init__(self, dirname, dataclass = DS, data_splits =  ["train", "val", "test"], ftype=".csv"):

        self.dirname = dirname
        self.data = None
        self.ftype = ftype
        self.data_splits = data_splits
        self.dataclass = dataclass

    def init_datasets(self, sets):

        for s in sets:
            #dynamically sets functions
            setattr(self,s, self.dataclass())
    
    def load_file(self, fname):
        pass

class Data(DS):


    def __init__(self):
        
        self.x = None 
        self.y = None

    def  __len__(self):
        pass 

    def __getitem__(self, idx):
        pass




    


    


        