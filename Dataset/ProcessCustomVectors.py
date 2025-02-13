import torch
import gensim
import string



def get_words(fname):
    words = []
    with open(fname, "r") as f:
        for row in f:
            word = row.strip("\n")
            words.append(word)
    words = [*string.punctuation, *words, *string.digits]
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

def main():
    
    fname = "./.raw_data/encodings/de.txt"
    vec_fname = "./.raw_data/encodings/de.vec"
    vec_save = "./.raw_data/encodings/de_custom.vec"

    words = get_words(fname)
    kv = create_embed_mapping(words, vec_fname)
    kv.save_word2vec_format(vec_save, binary=False)

if __name__ == "__main__":
    main()

    
