import torch
from torch import nn


device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    device = "mps"

# def beam_search(model, X, outlang, beam_width=5):
#     # (batch, seq_len, k)
#     init_out = torch.ones((X.shape[0], 1, 1)) * outlang.get_token("<SOS>")
#     init_out = init_out.long().to(device)
#     cands = []
#     k = 1
#     while (not (init_out == outlang.get_token("<EOS>")).any(dim=1).all()) and init_out.shape[1] < 100: 
#         init_save = init_out
#         seq_len = init_out.shape[1]
#         cands_prime = []
#         for i in range(len(k)):
#              # (batch, seq_len, num_tokens)
#             init_out = init_save[:,:,i]
#             probs = model.pred_prob(X, init_out)
#             # (batch, num_tokens)
#             probs = probs[:,-1]
#             #(batch, beam_width)
#             topk_values, topk_indices = torch.topk(probs, beam_width, dim=2)
#             cands_prime = []
#             for j in range(len(beam_width)):
#                 # (batch, seq_len, beam_width)
#                 init_out = init_out.unsqueeze(-1).expand(X.shape[0], seq_len, beam_width)
#                 # (batch, 1, beam_width)
#                 topk_indices = topk_indices.unsqueeze(1)
#                 topk_values = topk_values.unsqueeze(1)
#                 cand = torch.cat([init_out, topk_indices], dim=1)
#                 cands_prime.append(({"values":cand,"probs":topk_values}))
#             # merge cands_prime 


       
def get_first_eos_indices(tokens, eos_token):
    """
    Args:
        tokens (torch.Tensor): Tensor of shape (batch, seq_len, k)
        eos_token (int): The EOS token value.
    Returns:
        torch.Tensor: Tensor of shape (batch, k) containing the first index
                      where the EOS token appears for each beam. If no EOS is
                      found in a sequence, the returned index will be seq_len.
    """
    batch, seq_len, k = tokens.shape

    # Create a mask where True indicates the token equals eos_token.
    mask = (tokens == eos_token)

    # Create an indices tensor for the seq_len dimension.
    indices = torch.arange(seq_len, device=tokens.device).view(1, seq_len, 1).expand(batch, seq_len, k)

    # Where the EOS is not present, set the index to seq_len (a value larger than any valid index).
    indices_masked = torch.where(mask, indices, torch.tensor(seq_len, device=tokens.device))

    # The first occurrence of EOS along the sequence dimension is the minimum index.
    first_eos_indices = indices_masked.min(dim=1)[0]  # Shape: (batch, k)

    return first_eos_indices

    

def beam_search(model, X, outlang, beam_width=10):

    # (batch, seq_len, k)
    init_out = torch.ones((X.shape[0], 1, 1)) * outlang.get_token("<SOS>")
    init_out = init_out.long().to(device)
    prev_indx = None
    prev_values = None
    alpha = .25
    k = 1
    # print(outlang.embeddings.shape[0])
    while (not (init_out == outlang.get_token("<EOS>")).any(dim=1).all()) and init_out.shape[1] < 100: 
        
        seq_len = init_out.shape[1]
        # (batch, seq_len, k)
        init_out_save = init_out
        # (batch, seq_len * k, num_tokens)
        probs = model.pred_prob(X.unsqueeze(-1).expand(X.shape[0],X.shape[1],k).transpose(1,2).reshape((X.shape[0]*k, X.shape[1])), init_out.transpose(1,2).reshape((X.shape[0] *k , seq_len)))
        # print(init_out_save.shape)
        # (batch, seq_len, k, num_tokens)
        probs = probs.reshape((X.shape[0],k, seq_len, -1))
        probs = probs.transpose(1,2)
        # (batch, k, num_tokens)
        probs = probs[:,seq_len - 1]
        probs[:, :, outlang.get_token("<UNK>")] = 0
        probs = torch.clamp(probs, min=1e-10)
        probs = torch.log(probs)
        # print(probs.shape)
        #(batch, k, beam_width)
        topk_values, topk_indices = torch.topk(probs, beam_width, dim=2, sorted=True,largest=True)
        # print("topk:",topk_indices)
        # print(topk_values)
        # (batch, seq_len, k * beam_width)
        init_out = init_out_save
        candidates = init_out.unsqueeze(-1).expand(init_out.shape[0], seq_len,k, beam_width)
        topk_indices = topk_indices.unsqueeze(1)
        candidates = torch.cat([candidates, topk_indices], dim=1)
        candidates = candidates.reshape(init_out.shape[0],seq_len + 1, k * beam_width)
        eos = get_first_eos_indices(init_out_save, outlang.get_token("<EOS>"))
        eos = eos.unsqueeze(1).expand(X.shape[0],k, beam_width)
        if prev_values is not None:
            # don't add new values if sentence contains eos
            # eos = eos_values.unsqueeze(1).expand(X.shape[0],beam_width, beam_width)
            topk_values = prev_values + topk_values * (eos == seq_len).long()
            # print(topk_values.shape)
            # prev_values = topk_values
        # (batch, k * beam_width)
        topk_values = topk_values * (1.0/(eos.float())**alpha)
        topk_values = topk_values.reshape((X.shape[0],k * beam_width))
        # print(topk_values.shape)
        values, topk_indices = torch.topk(topk_values, beam_width, dim=1, sorted=True,largest=True)
        topk_indices = topk_indices.unsqueeze(1).expand(X.shape[0], candidates.shape[1], beam_width).long()
        prev_values = values.unsqueeze(-1).expand((X.shape[0], beam_width, beam_width))
        candidates = torch.gather(candidates, 2, index=topk_indices)
        init_out = candidates
        # (batch, seq_len, beam_width)
        # print("mine:",prev_values)
        prev_values = model.compute_p_sentence(X.unsqueeze(-1).expand(*X.shape, beam_width).transpose(1,2).reshape((X.shape[0]* beam_width, X.shape[1])),init_out.transpose(1,2).reshape((init_out.shape[0]* beam_width, (seq_len + 1))), eos_token = outlang.get_token("<EOS>"))
        prev_values = prev_values.reshape((X.shape[0], beam_width)).unsqueeze(-1).expand((X.shape[0], beam_width, beam_width))
        # print("fuck face:",prev_values)
        k = beam_width
    # print(prev_values[0,0,0])
    init_out = init_out[:,:,0]
    exit_out = torch.ones((X.shape[0], 1)) * outlang.get_token("<EOS>")
    exit_out = exit_out.long()
    init_out = torch.concat((init_out, exit_out.to(device)), dim = 1)
    return init_out
        



def greedy_search(model, X, outlang):

    #generate init out starting with start of sentence token
    init_out = torch.ones((X.shape[0], 1)).to(device) * outlang.get_token("<SOS>")
    # while eos not in all sentences keep generating
    while not (init_out == outlang.get_token("<EOS>")).any(dim=1).all():
        # predict the probs and grab the top token
        init_out = init_out.long()
        probs = model.pred_prob(X, init_out)
        indicies = probs.argmax(dim=2)
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
