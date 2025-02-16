from .Language import Language, get_language_loader
import os

class EngToFr(Language):


    def __init__(self):
        
        super().__init__(os.path.join("wmt","eng_to_fr"),"en","fr",True) 

class FrToEng(Language):


    def __init__(self):
        
        super().__init__(os.path.join("wmt","eng_to_fr"),"fr","en",False) 

if __name__ == "__main__":

    dataset = EngToFr()
    dataset.test_init()
    dataset.val_init()
    dataset.train_init()
    
    print(len(dataset.train))
    print(dataset.inlang.embeddings.shape)
    print(dataset.outlang.embeddings.shape)
    # d = get_language_loader(dataset.test)
    # for x,y in d:
    #     # print(f"\n\n{x}\n\n")
    #     print(x.shape)
        
        
