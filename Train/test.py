import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    device = "mps"

def test_model(model, dataset,loss, fname,search=None, metrics=[]):

    model.load(fname)
    l_avg, bleu, acc =  model.test(dataset,loss,search, metrics)
    l_avg.display()
    bleu.display()
    acc.display()


def infer(model, dataset, fname, search):

    model.load(fname)
    model.transformer.eval()
    while True:
        with torch.no_grad():
            print("Please enter a sentence:",end="")
            sentence = input()
            tok_sentence = dataset.inlang.sentence_to_tok(sentence)
            tok_sentence = tok_sentence.reshape((-1, *tok_sentence.shape))
            print(tok_sentence)
            out = search(model, tok_sentence.to(device), dataset.outlang, beam_width=10)
            print(f"Output:{dataset.decode_sentence(out[0].cpu())}")