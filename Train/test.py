

def test_model(model, dataset,loss, fname, metrics=[]):

    model.load(fname)
    model.test(dataset,loss, metrics)
