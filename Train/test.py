

def test_model(model, dataset,loss, fname,search=None, metrics=[]):

    model.load(fname)
    l_avg, bleu, acc =  model.test(dataset,loss,search, metrics)
    l_avg.display()
    bleu.display()
    acc.display()
