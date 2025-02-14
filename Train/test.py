

def test_model(model, dataset,loss, fname, metrics=[]):

    model.load(fname)
    l_avg, bleu, acc =  model.test(dataset,loss, metrics)
    l_avg.display()
    bleu.display()
    acc.display()