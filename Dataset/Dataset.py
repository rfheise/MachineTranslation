

class Dataset():

    def __init__(self, fname, ftype=".csv"):

        self.fname = fname 
        self.data = None
        self.data_splits = ["train","test","val"]
        self.ftype = ftype

        #initis splits to None
        for split in self.data_splits:
            setattr(self, split, None)
        self.init_datasets(self.data_splits)

    def init_datasets(self, sets):

        for s in sets:
            #initializes split functions
            def lmb(self):
                # calls load fun i.e. load_train if split is None
                if getattr(self, s) is None:
                    getattr(self,f"{s}_load")()
                # sets data to be loaded dataset
                self.data = getattr(self, f"{s}_data")

            def lmb_load(self):
                #calls load function and updates corresponding split
                setattr(self, f"{s}_data",self.load_data(s))

            #dynamically sets functions
            setattr(self,s, lmb)
            setattr(self,s, lmb_load)
    
    def load_data(self, split):
        return self.load_file(f"{self.fname}-{split}.{self.ftype}")
    
    def load_file(self, fname):
        pass

class Data():


    def __init__(self, data_x, data_y):

        self.x = data_x 
        self.y = data_y





    


    


        