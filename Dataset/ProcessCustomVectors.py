import torch
import gensim
import string
import regex
import csv
import os

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

def generate_custom_vec(dataset_folder, lang):
    
    fname = os.path.join(dataset_folder, f"{lang}.txt")
    vec_fname = os.path.join(dataset_folder,f"{lang}.vec")
    vec_save = os.path.join(dataset_folder,f"{lang}_custom.vec")

    words = get_words(fname)
    kv = create_embed_mapping(words, vec_fname)
    kv.save_word2vec_format(vec_save, binary=False)

def get_unique_words(dataset_folder, lang, col):
    pattern = regex.compile(r'\p{L}+|\d|[^\w\s]')
    lang1 = set()
    first_row = True
    with open(os.path.join(dataset_folder, "val.csv"), "r") as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            if first_row:
                first_row = False 
                continue
            if len(row) != 2:
                continue
            for word in pattern.findall(row[col]):
                    lang1.add(word.lower())
    
    with open(os.path.join(dataset_folder, f"{lang}.txt"), "a") as f:
        for word in lang1:
            f.write(word + "\n")
        

if __name__ == "__main__":
    # langs = ["en","de"]
    langs = [{"lang":"en","col":0},{"lang":"fr","col":1}]
    dataset_folder = "./.raw_data/wmt/eng_to_fr"
    for lang in langs:
        print(f"getting unique words for {lang['lang']}")
        get_unique_words(dataset_folder, lang["lang"], lang["col"])
        print(f"creating embeddings for {lang['lang']}")
        generate_custom_vec(dataset_folder,lang["lang"])

    
