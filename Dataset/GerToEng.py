from .Language import Language, get_language_loader
import os

class EngToGer(Language):


    def __init__(self, byte_pair = False):
        
        super().__init__(os.path.join("wmt","ger_to_eng_byte_pair"),"en","de",True, byte_pair) 

class GerToEng(Language):


    def __init__(self, byte_pair = False):
        
        super().__init__(os.path.join("wmt","ger_to_eng_byte_pair"),"de","en",False, byte_pair) 

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
        
        
