from .Dataset import Dataset, Data
import csv 
import os
import torch
import gensim
from torch.utils.data import DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
import regex
import sqlite3
import numpy as np
import pandas as pd 
import random
import multiprocessing
from ..Log import Logger 
import sentencepiece as spm
device = Logger.device

class Embeddings():

    special_toks = ["<PAD>","<SOS>", "<EOS>", "<UNK>"]
    special_toks_val = None
    embedding_dim = 300
    pattern = regex.compile(r'\p{L}+|\d|[^\w\s]')

    def __init__(self, dname, lang):
        self.dname = dname
        self.lang = lang
        self.fname = os.path.join(dname, lang+"_custom.vec")
        self.load_embeddings()
        self.embeddings = None

    def sentence_to_tok(self, sentence):
        toks = Embeddings.pattern.findall(sentence)
        toks = ["<SOS>",*toks, "<EOS>"]
        toks = torch.tensor([int(self.get_token(tok)) for tok in toks])
        return toks
    def get_token_count(self):
        return self.embeddings.shape[0]
    
    def get_embed_dim(self):
        return self.embeddings.shape[1]
    
    def load_embeddings(self):

        self.ft_model =  gensim.models.KeyedVectors.load_word2vec_format(self.fname, binary=False)
        self.vocab = [*self.ft_model.index_to_key]
        self.vocab.sort()
        self.vocab = [*self.special_toks, *self.vocab]
        self.word_to_tok = {word: ix for ix, word in enumerate(self.vocab)}
        self.tok_to_word = {ix: word for ix, word in enumerate(self.vocab)}
        self.embeddings = self.create_embeddings()
        Embeddings.special_toks_val = {tok: ix for ix, tok in enumerate(Embeddings.special_toks)}

    def create_embeddings(self):
        embeddings = torch.randn((len(self.vocab), 300))
        for i, word in enumerate(self.vocab):
            if word in self.ft_model.key_to_index:
                embeddings[self.word_to_tok[word]] = torch.tensor(self.ft_model.get_vector(word))
        return embeddings

    def get_token(self, word):
        if word in self.special_toks:
            return self.word_to_tok[word]
        word = word.lower()
        if word not in self.vocab:
            return self.word_to_tok["<UNK>"]
        
        return self.word_to_tok[word]

    def convert_row_to_tensor(self, pattern, row):
        toks = pattern.findall(row)
        toks = ["<SOS>",*toks, "<EOS>"]
        toks = [int(self.get_token(tok)).to_bytes(4, byteorder='little') for tok in toks]
        tok_bytes = b''.join(toks)
        return tok_bytes
    
    def get_word(self, tok):
        return self.tok_to_word[int(tok)]
    
    def get_embeddings(self, tok):
        return self.embeddings[tok]
    
class BytePairEmbeddings(Embeddings):

    def __init__(self, dname, lang, vocab_size=37000):
        self.vocab_size = vocab_size
        super().__init__(dname, lang)

    def load_embeddings(self):
        tok_file = os.path.join(self.dname, "lang_embed.model")
        if not os.path.exists(tok_file):
            self.generate_byte_pair_encoding()
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(tok_file)
        Embeddings.special_toks_val = {ix:int(self.get_token(ix)) for ix in Embeddings.special_toks}

    def get_token_count(self):
        return self.tokenizer.get_piece_size()
    
    def get_embed_dim(self):
        return None
    
    def convert_row_to_tensor(self, pattern, row):
        toks = row
        # print(toks)
        # print(self.sentence_to_tok(toks))
        toks = [int(tok).to_bytes(4, byteorder='little') for tok in self.sentence_to_tok(toks)]
        tok_bytes = b''.join(toks)
        return tok_bytes
    
    def generate_byte_pair_encoding(self):
        dataset = "train"
        tmp_file = os.path.join(self.dname, dataset+".txt")
        print("generating tmp output text file")
        with open(os.path.join(self.dname, dataset+".csv"), "r") as f:
            with open(tmp_file, "w") as f2:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if len(row) != 2:
                        continue
                    for item in row:
                        item = item.strip("\n").strip()
                        f2.write(item + "\n")
        print("generating byte pair encoding")           
        spm.SentencePieceTrainer.train(
            input=tmp_file,              # Training data file
            model_prefix=os.path.join(self.dname, "lang_embed"),              # Output file prefix
            vocab_size=self.vocab_size,               # Desired vocabulary size
            character_coverage=1.0,        # Full character coverage for English
            model_type='bpe',              # Use Byte-Pair Encoding
            input_sentence_size=1000000,   # Use up to 1,000,000 sentences from the data
            shuffle_input_sentence=True,   # Shuffle sentences to reduce bias
            user_defined_symbols=Embeddings.special_toks,  # Additional symbols to include
            bos_piece = "<SOS>",
            eos_piece = "<EOS>"
        )
        print("removing tmp output text file")
        os.remove(tmp_file)
        print("finished generating byte pair encoding")

    def create_embeddings(self):
        return None
    
    def get_embeddings(self, tok):
        return None

    def sentence_to_tok(self, sentence):
        return [self.get_token(tok) for tok in self.tokenizer.encode_as_pieces(sentence)]

    def get_word(self, tok):
        return self.tokenizer.decode_ids([int(tok),])

    def get_token(self, word):
        return self.tokenizer.piece_to_id(word)
    
    def get_sentence(self, toks):
        return self.tokenizer.decode_ids([int(tok) for tok in toks])


    

class Language(Dataset):

    def __init__(self, dirname,Embeddings=Embeddings, inlang="de", outlang="en", flip=False, byte_pair = False):
        current_file_dir = os.path.split(os.path.abspath(__file__))[0]
        dirname = os.path.join(current_file_dir,".raw_data", dirname)
        self.byte_pair = byte_pair
        super().__init__(dirname, LanguageData)
        if self.byte_pair:
            self.inlang = Embeddings(dirname, inlang)
            self.outlang =  Embeddings(dirname, outlang)
        else:
            self.inlang = BytePairEmbeddings(dirname, inlang)
            self.outlang = self.inlang
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

    def is_single_embeddings(self):
        return not self.byte_pair
    
    def decode_sentence(self, sentence, lang="outlang", as_string=True):
        if lang == "outlang":
            lang = self.outlang 
        else:
            lang = self.inlang 
        try:
            return "".join(lang.get_sentence(sentence))
        except:
            pass
        toks = []
        for tok in sentence:
            word = lang.get_word(tok)
            if word == "<EOS>":
                break
            if word in Embeddings.special_toks and word != "<UNK>":
                continue
            toks.append(word)
        if as_string:
            return " ".join(toks)
        return toks
    
    def decode_sentences(self, sentences, lang="outlang"):
        s = []
        for sentence in sentences:
            s.append(self.decode_sentence(sentence, lang, True))
        return s
    
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
        self.pool_size = 24
        self.pattern = regex.compile(r'\p{L}+|\d|[^\w\s]')
    def load_data(self):
        if not os.path.exists(self.db_path):
            print(self.db_path)
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
                    if counter % (10000 * self.pool_size) == 0 and len(cache) != 0:
                        cache = self.process_cache(cache)
                        self.insert_many(cache)
                        cache = []
                        print(counter)
                    cache.append(row)
                    counter += 1
            if len(cache) != 0:
                cache = self.process_cache(cache)
                self.insert_many(cache)
            self.conn.close()
            self.conn = None 
            self.c = None
            
    
    def process_cache(self, rows):

        cache = []
        processes = []
        manager = multiprocessing.Manager()
        shared_cache = manager.list()
        for i in range(self.pool_size):
            start = (len(rows)//self.pool_size + 1) * i 
            end = (len(rows)//self.pool_size + 1) * (i + 1)
            if self.flip:
                p = multiprocessing.Process(target=LanguageData.single_proc, args=(shared_cache, rows[start:end], self.outlang, self.inlang, self.pattern))
            else:
                p = multiprocessing.Process(target=LanguageData.single_proc, args=(shared_cache, rows[start:end], self.inlang, self.outlang, self.pattern))
            
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        ret = list(shared_cache)
        return ret

    @staticmethod
    def single_proc(q, rows, inlang, outlang, pattern):
        cache = []
        for row in rows:
            in_lang_toks = LanguageData.convert_row_to_tensor(pattern, row[0], inlang)
            out_lang_toks = LanguageData.convert_row_to_tensor(pattern, row[1], outlang)
            cache.append((row[0], in_lang_toks, row[1], out_lang_toks))
        q.extend(cache)

    @staticmethod
    def convert_row_to_tensor(pattern, row, lang):
        return lang.convert_row_to_tensor(pattern, row)
        

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
        if isinstance(idx, (list, slice)):
        # Convert to a list if it's a slice
            if isinstance(idx, slice):
                idx = list(range(*idx.indices(len(self))))
            return [self.__getitem__(i) for i in idx]
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
        return torch.tensor(np.frombuffer(row, dtype=np.int32, count=len(row)//4)).long()

    
    def __delete__(self):
        if self.conn is not None:
            self.conn.close()

def collate_fn_gen(token_limit):
    def pad_collate_fn(batch):

        # try:
        #     xs, ys = zip(*batch[0])
        #     batch = batch[0]
        # except:
        #     xs, ys = zip(*batch)
        #     batch = batch
        items = []
        # print(batch)
        for item in batch:
            if len(item[0]) > token_limit or len(item[1]) > token_limit:
                continue 
            items.append(item)
        batch = items
        xs, ys = zip(*batch)
        # Pad the list of tensors to create a single tensor of shape [batch_size, max_seq_len]
        xs_padded = pad_sequence(xs, batch_first=True, padding_value=Embeddings.special_toks_val["<PAD>"])
        ys_padded = pad_sequence(ys, batch_first=True, padding_value=Embeddings.special_toks_val["<PAD>"])
        return xs_padded.long(), ys_padded.long()
    return pad_collate_fn

def pad_collate_fn(batch):

        # try:
        #     xs, ys = zip(*batch[0])
        #     batch = batch[0]
        # except:
        #     xs, ys = zip(*batch)
        #     batch = batch
        items = []
        # print(batch)
        for item in batch:
            if len(item[0]) > 100 or len(item[1]) > 100:
                continue 
            items.append(item)
        batch = items
        xs, ys = zip(*batch)
        # Pad the list of tensors to create a single tensor of shape [batch_size, max_seq_len]
        xs_padded = pad_sequence(xs, batch_first=True, padding_value=Embeddings.special_toks_val["<PAD>"])
        ys_padded = pad_sequence(ys, batch_first=True, padding_value=Embeddings.special_toks_val["<PAD>"])
        return xs_padded.long(), ys_padded.long()

def language_loader_init_fn(worker_id):
    """Initialize a database connection for each worker."""
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset  # Access the dataset object
        dataset.load_data()  # Initialize the SQLite connection

def get_language_loader(dataset, token_limit= 100, batch_size=64, shuffle=True,num_workers = 12, worker_init_fn=language_loader_init_fn):
    #return DataLoader(dataset, batch_size=batch_size, num_workers = num_workers,shuffle=shuffle, worker_init_fn=worker_init_fn,collate_fn=pad_collate_fn)
    # loader = DataLoader(dataset, sampler=TokenBatchSampler(dataset, token_limit, shuffle), num_workers = num_workers, worker_init_fn=worker_init_fn,collate_fn=collate_fn)
    if device == "mps":
        return  DataLoader(dataset, batch_size=batch_size, num_workers = num_workers//2,shuffle=shuffle, worker_init_fn=worker_init_fn,collate_fn=pad_collate_fn,pin_memory=True)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers = num_workers,shuffle=shuffle, worker_init_fn=worker_init_fn,collate_fn=collate_fn_gen(token_limit),pin_memory=True)
    return loader

        
        

class TokenBatchSampler(Sampler[int]):

    def __init__(self, dataset, token_limit, shuffle=True):
        """
        Custom sampler that groups samples into batches by token count.

        Args:
            dataset (Dataset): Your dataset.
            token_limit (int): Maximum number of tokens per batch.
            shuffle (bool): Whether to shuffle the indices at the start of each iteration.
        """
        self.dataset = dataset
        self.token_limit = token_limit
        self.shuffle = shuffle
        self.approx = 1000

    def __iter__(self):
        indxs = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indxs)
        
        batch = []
        sentence_count = 0
        max_len = 0

        for index in indxs:
            
            sentence = self.dataset[index]
            length = self.get_indv_len(sentence)
            if length > self.token_limit:
                # continue
                pass
            
            
            # If adding this sentence would exceed the token limit,
            # yield the current batch (if not empty) and start a new one.
            if length > max_len:
                max_len = length

            if (sentence_count + 1) * max_len > self.token_limit and len(batch) > 0:
                yield batch
                batch = []
                sentence_count = 0
                max_len = 0
            
            if length > max_len:
                max_len = length

            batch.append(index)
            sentence_count +=  1
            
        
        if batch:
            yield batch

    def get_indv_len(self, sentence):
        return len(sentence[0]) + len(sentence[1])
    
    def __len__(self):

        #approximate length using first 1000 sentences
        val = 0
        sentences = min(len(self.dataset),self.approx)
        max_length = 0
        for i in range(sentences):
            length = self.get_indv_len(self.dataset[i])
            if length > max_length:
                max_length = length
            
        return round(max_length * len(self.dataset))//self.token_limit




if __name__ == "__main__":

    dataset = Language(os.path.join("eng_to_ger"),Embeddings=BytePairEmbeddings)
    dataset.train_init()
    print(len(dataset.test))
    d = get_language_loader(dataset.test)
    for x,y in d:
        print(f"\n\n{y}\n\n")
