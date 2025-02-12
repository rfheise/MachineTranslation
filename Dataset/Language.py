from .Dataset import Dataset, Data
import csv 
import os
import torch
import gensim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import re
import sqlite3
import numpy as np
import pandas as pd 

class Embeddings():

    special_toks = ["<PAD>","<SOS>", "<EOS>", "<UNK>"]
    special_toks_val = {tok: ix for ix, tok in enumerate(special_toks)}
    embedding_dim = 300

    def __init__(self, fname):
        self.fname = fname 
        self.load_embeddings()

    def load_embeddings(self):

        ft_model =  gensim.models.KeyedVectors.load_word2vec_format(self.fname, binary=False, limit=10000)
        self.vocab = [*ft_model.index_to_key]
        self.vocab.sort()
        self.vocab = [*self.special_toks, *self.vocab]
        self.word_to_tok = {word: ix for ix, word in enumerate(self.vocab)}
        self.tok_to_word = {ix: word for ix, word in enumerate(self.vocab)}
        pre_trained = torch.tensor(ft_model.vectors, dtype=torch.float32)
        random_vectors = torch.randn(len(Embeddings.special_toks), Embeddings.embedding_dim)
        self.embeddings = torch.cat((random_vectors, pre_trained), dim=0)

    def get_token(self, word):
        if word not in self.vocab:
            return self.word_to_tok["<UNK>"]
        
        return self.word_to_tok[word]
    
    def get_word(self, tok):
        return self.tok_to_word[tok]
    
    def get_embeddings(self, tok):
        return self.embeddings[tok]
    

class Language(Dataset):

    def __init__(self, dirname, inlang="de", outlang="en", flip=False):
        current_file_dir = os.path.split(os.path.abspath(__file__))[0]
        dirname = os.path.join(current_file_dir,".raw_data", dirname)
        super().__init__(dirname, LanguageData)
        self.inlang = Embeddings(os.path.join(current_file_dir,".raw_data", "encodings",inlang+".vec"))
        self.outlang =  Embeddings(os.path.join(current_file_dir,".raw_data", "encodings",outlang+".vec"))
        self.flip = flip
        self.init_datasets(self.data_splits) 
        
    
    def init_datasets(self, sets):

        for s in sets:
            setattr(self,s, LanguageData(self.get_fname(s), self.inlang, self.outlang, self.flip))
            def lmbload(s=s):
                getattr(self,s).load_data()
            def lmbload(s=s):
                getattr(self,s).init_data()
            setattr(self, s + "_load", lmbload)
            setattr(self, s + "_init", lmbload)

                
    def get_fname(self, split):
        return os.path.join(self.dirname, split)
    
                                
class LanguageData(Data):

    def __init__(self, fname, inlang, outlang, flip):
        super().__init__()
        self.fname = fname
        self.inlang = inlang
        self.outlang = outlang
        self.db_path = self.fname + ".sqlite3"
        self.csv_path = self.fname + '.csv'
        self.conn = None
        self.c = None
        self.flip = flip

    def load_data(self):
        if not os.path.exists(self.db_path):
            exit("Dataset not Initialized!!!!")
        if self.conn is None or self.c is None:
            self.conn = sqlite3.connect(self.db_path)
            self.c = self.conn.cursor()

    def create_sql_table(self):

        self.c.execute("CREATE TABLE IF NOT EXISTS translations (id INTEGER PRIMARY KEY,in_lang_text TEXT NOT NULL,in_lang_bin BLOB,out_lang_text TEXT NOT NULL,out_lang_blob BLOB);")
        self.conn.commit()

    def insert_one(self,item):
        self.c.execute("""
        INSERT INTO translations (in_lang_text, in_lang_bin, out_lang_text, out_lang_blob)
        VALUES (?, ?, ?, ?)
    """, item)
        self.conn.commit()

    def insert_many(self,items):
        self.c.executemany("""
        INSERT INTO translations (in_lang_text, in_lang_bin, out_lang_text, out_lang_blob)
        VALUES (?, ?, ?, ?)
    """, items)
        self.conn.commit()
        exit()

    def init_data(self):
        if not os.path.exists(self.db_path):
            self.conn = sqlite3.connect(self.db_path)
            self.c = self.conn.cursor()
            self.create_sql_table()
            counter = 0
            cache = []
            with open(self.csv_path, "r") as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if len(row) != 2:
                        continue
                    if counter % 10000 == 0 and len(cache) != 0:
                        print(counter)
                        self.insert_many(cache)
                        cache = []
                    in_lang_toks = self.convert_row_to_tensor(row[0], self.inlang)
                    out_lang_toks = self.convert_row_to_tensor(row[1], self.outlang, counter % 10000 == 0)
                    cache.append((row[0], in_lang_toks, row[1], out_lang_toks))
                    counter += 1
            if len(cache) != 0:
                self.insert_many(cache)
            self.conn.close()
            self.conn = None 
            self.c = None

    def convert_row_to_tensor(self, row, lang, p = False):

        pattern = r"\w+|[^\w\s]"
        toks = ["<SOS>",*re.findall(pattern, row), "<EOS>"]
        if p:
            print()
            print()
            for i, tok in enumerate(toks):
                print(f"{tok}:{lang.get_word(lang.get_token(tok))}, ", end="")
            print()
            print()
        toks = [int(lang.get_token(tok)).to_bytes(4, byteorder='little') for tok in toks]
        tok_bytes = b''.join(toks)
        return tok_bytes

    def __len__(self):
        if self.conn is None:
            self.load_data()
        # Open a connection and fetch the count of rows in the table
        self.c.execute("SELECT COUNT(*) FROM translations")
        count = self.c.fetchone()[0]
        self.conn.close()
        self.c = None
        self.conn = None
        return count

    def __getitem__(self, idx):
        if self.c is None or self.conn is None:
            self.load_data()
        # SQLite indices usually start at 1, so adjust if necessary
        db_idx = idx + 1

        # Fetch the row with the corresponding ID
        self.c.execute("SELECT * FROM translations WHERE id = ?", (db_idx,))
        row = self.c.fetchone()
        
        # Return just the row or process it further if needed
        if self.flip:
            return self.convert_to_token_arr(row[4]), self.convert_to_token_arr(row[2])
        else:
            return self.convert_to_token_arr(row[2]), self.convert_to_token_arr(row[4])

    def convert_to_token_arr(self, row):
        return torch.tensor(np.frombuffer(row, dtype=np.uint32, count=len(row)//4)).long()

    
    def __delete__(self):
        if self.conn is not None:
            self.conn.close()


def pad_collate_fn(batch):
    
    xs, ys = zip(*batch)  # Unzip the batch of (x, y) pairs.
    
    # Pad the list of tensors to create a single tensor of shape [batch_size, max_seq_len]
    xs_padded = pad_sequence(xs, batch_first=True, padding_value=Embeddings.special_toks_val["<PAD>"])
    ys_padded = pad_sequence(ys, batch_first=True, padding_value=Embeddings.special_toks_val["<PAD>"])

    return xs_padded, ys_padded

def language_loader_init_fn(worker_id):
    """Initialize a database connection for each worker."""
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset  # Access the dataset object
        dataset.load_data()  # Initialize the SQLite connection

def get_language_loader(dataset, batch_size=128, shuffle=True,num_workers = 4, worker_init_fn=language_loader_init_fn,collate_fn=pad_collate_fn):

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,num_workers = num_workers, worker_init_fn=worker_init_fn,collate_fn=collate_fn)
    return loader
if __name__ == "__main__":

    dataset = Language(os.path.join("wmt","eng_to_ger"))
    dataset.test_init()
    print(len(dataset.test))
    d = get_language_loader(dataset.test)
    for x,y in d:
        print(f"\n\n{y}\n\n")
        
        
