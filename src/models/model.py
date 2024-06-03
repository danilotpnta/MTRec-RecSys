import torch
import torch.nn as nn

from transformers import BertModel, AutoModel

class UserEmbedding(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.W = nn.Linear(hidden, hidden)
        self.q = nn.Parameter(torch.randn(hidden))

    def forward(self, X):
        att = self.q * torch.tanh(self.W(X))
        att_weight = att.softmax(dim=-1)
        
        feature = torch.sum(X * att_weight, dim = 1)
        return feature
    
class RankingScore(nn.Module):
    def __init__(self, pretrained, hidden, only_feature=True):
        super().__init__()
        self.news_encoder = AutoModel.from_pretrained(pretrained)
        self.user_encoder = UserEmbedding(hidden)
        self.only_feature = only_feature

    def forward(self, hist, mask_hist, cand, mask_cand, bz=1):
        '''
            samples:  BN(BM) x L x H
        '''

        if not self.only_feature:
            hist_coding = self.encode_news(hist, mask_hist) # BN x H
            cand_coding = self.encode_news(cand, mask_cand) # BM x H
        else:
            with torch.no_grad():
                hist_coding = self.encode_news(hist, mask_hist) # BN x H
                cand_coding = self.encode_news(cand, mask_cand) # BM x H
            
            # hist_coding = hist_coding.detach()
            # cand_coding = cand_coding.detach()
        
        hist_coding = self.reshape(hist_coding, bz) # B x N 
        cand_coding = self.reshape(cand_coding, bz) # B x M x H
        user_coding = self.user_encoder(hist_coding) # B x H
        score = torch.bmm(cand_coding, user_coding.unsqueeze(-1)) # B x M x 1
        return score

    def encode_news(self, news, mask):
        # print(news.shape, mask.shape)

        feature = self.news_encoder(news, mask)
        return feature['last_hidden_state'][:, 0]

    def reshape(self, batch_news, bz):
        n_news = len(batch_news) // bz
        reshaped_batch = batch_news.reshape(bz, n_news, -1)
        return reshaped_batch
