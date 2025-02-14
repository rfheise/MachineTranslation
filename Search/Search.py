import torch


device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    device = "mps"


def beam_search(model, X, outlang, beam_width=5):

    init_out = torch.ones((input.shape[0], 1)) * outlang.get_token("<SOS>")
    probs = model.pred_proba(X, init_out)
    topk_values, topk_indices = torch.topk(probs, beam_width, dim=1)

    while True: 

        pass 

def greedy_search(model, X, outlang):

    #generate init out starting with start of sentence token
    init_out = torch.ones((X.shape[0], 1)).to(device) * outlang.get_token("<SOS>")
    # while eos not in all sentences keep generating
    while not (init_out == outlang.get_token("<EOS>")).any(dim=1).all():
        # predict the probs and grab the top token
        init_out = init_out.long()
        probs = model.pred_prob(X, init_out)
        print(probs)
        print(probs.shape)
        indicies = probs.argmax(dim=2)
        print(indicies)
        print(indicies.shape)
        exit()
        # print(indicies)
        init_out = torch.concat((init_out, indicies), dim=1)
        # if more than 100 tokens have been predicted without EOS it's too long
        if init_out.shape[1] > 100:
            break
    # add eos to end all sentences not finished
    exit_out = torch.ones((X.shape[0], 1)) * outlang.get_token("<EOS>")
    init_out = torch.concat((init_out, exit_out.to(device)), dim = 1)
    # exit()
    return init_out
