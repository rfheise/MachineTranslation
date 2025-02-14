from .Language import Language, get_language_loader
import os

class EngToGer(Language):


    def __init__(self):
        
        super().__init__(os.path.join("wmt","eng_to_ger"),"en","de",True) 

class GerToEng(Language):


    def __init__(self):
        
        super().__init__(os.path.join("wmt","eng_to_ger"),"de","en",False) 

if __name__ == "__main__":

    dataset = EngToGer()
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
        
        
