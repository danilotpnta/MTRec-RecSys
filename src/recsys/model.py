import torch
from torch import nn
from torch.nn import functional as F

class MTRec(nn.Module):
    """The main prediction model for the multi-task recommendation system, as described in the paper by ...
    """
    def __init__(self, hidden_dim):
        super(MTRec, self).__init__()

        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.q = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, history, candidates):
        '''
            B - batch size (keep in mind we use an unusual mini-batch approach)
            H - history size (number of articles in the history, usually 30)
            D - hidden size (768)
            history:    B x H x D 
            candidates: B x 1 x D
        '''

        # print(f"{candidates.shape=}")
        att = self.q * F.tanh(self.W(history))
        att_weight = F.softmax(att, dim=1)
        # print(f"{att_weight.shape=}")

        user_embedding = torch.sum(history * att_weight, dim = 1)
        # print(f"{user_embedding.shape=}")
        # print(f"{user_embedding.unsqueeze(-1).shape=}")
        score = torch.bmm(candidates, user_embedding.unsqueeze(-1)) # B x M x 1
        # print(score.shape)
        return score.squeeze(-1)

    def reshape(self, batch_news, bz):
        n_news = len(batch_news) // bz
        reshaped_batch = batch_news.reshape(bz, n_news, -1)
        return reshaped_batch