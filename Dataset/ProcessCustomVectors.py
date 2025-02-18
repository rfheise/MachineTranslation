import torch
import gensim
import string
import regex
import csv

def get_words(fname):
    words = set()
    with open(fname, "r") as f:
        for row in f:
            word = row.strip("\n")
            words.add(word)
    for tok in [*string.punctuation, *string.digits]:
         words.add(tok)
    words = [*words]
    words.sort()
    return words

def create_embed_mapping(words, vec_fname):

    d = {}
    ft_model = gensim.models.KeyedVectors.load_word2vec_format(vec_fname, binary=False)
    for word in words:
        if word in ft_model.key_to_index:
            d[word] = ft_model.get_vector(word)
    kv = gensim.models.KeyedVectors(len(d[[*d.keys()][0]]))
    kv.add_vectors(list(d.keys()), list(d.values()))
    print(len(d.keys()))
    return kv

def generate_custom_vec(lang):
    
    fname = f"./.raw_data/encodings/{lang}.txt"
    vec_fname = f"./.raw_data/encodings/{lang}.vec"
    vec_save = f"./.raw_data/encodings/{lang}_custom.vec"

    words = get_words(fname)
    kv = create_embed_mapping(words, vec_fname)
    kv.save_word2vec_format(vec_save, binary=False)

def get_unique_words(fname, lang, col):
    pattern = regex.compile(r'\p{L}+|\d|[^\w\s]')
    lang1 = set()
    with open(fname, "r") as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            if len(row) != 2:
                continue
            for word in pattern.findall(row[col]):
                    lang1.add(word.lower())
    with open(f"./.raw_data/encodings/{lang}.txt", "a") as f:
        for word in lang1:
            f.write(word + "\n")
        

if __name__ == "__main__":
    # langs = ["en","de"]
    langs = ['fr']
    for lang in langs:
        print(f"getting unique words for {lang}")
        get_unique_words("./.raw_data/wmt/eng_to_fr/val.csv", lang)
        print(f"creating embeddings for {lang}")
        generate_custom_vec(lang)

    
