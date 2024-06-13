import torch
from torch import nn
from torch.nn import functional as F

import torch.nn.functional as F

def apply_softmax_crossentropy(logits, one_hot_targets, epsilon=1e-10):
    """
    Applies softmax and computes the cross-entropy loss for each segment of logits with one-hot encoded targets.

    Args:
        logits (torch.Tensor): Flattened array of logits.
        repeats (torch.Tensor): Tensor indicating the number of logits in each segment.
        one_hot_targets (torch.Tensor): Flattened one-hot encoded target labels.
        epsilon (float): A small value to add to log inputs to avoid NaN.

    Returns:
        torch.Tensor: The cross-entropy loss for each segment.
    """

    # Split logits and one-hot targets according to repeats
    #split_logits = torch.split(logits, repeats.tolist())
    #split_targets = torch.split(one_hot_targets, repeats.tolist())

    # Determine the maximum length for padding
    #max_len = max(repeats)
    
    # Pad logits and one-hot targets, then stack them
    #padded_logits = torch.stack([F.pad(segment, (0, max_len - segment.size(0)), 'constant', float('-inf')) for segment in split_logits])
    #padded_targets = torch.stack([F.pad(segment, (0, max_len - segment.size(0)), 'constant', 0) for segment in split_targets])
    
    # Apply softmax to logits and add epsilon to avoid NaNs
    softmaxed_logits = F.softmax(logits, dim=-1)
    log_softmaxed_logits = torch.log(softmaxed_logits + epsilon)
    
    # Calculate cross-entropy loss
    losses = -torch.sum(one_hot_targets * log_softmaxed_logits, dim=-1)
    
    # Mask out the padded positions
    mask = (one_hot_targets.sum(dim=-1) > 0).float()
    masked_losses = losses * mask
    
    # Sum the losses for each segment and divide by the number of true (non-padded) entries
    segment_losses = masked_losses.sum(dim=-1) / mask.sum(dim=-1)

    return segment_losses


class MTRec(nn.Module):
    """The main prediction model for the multi-task recommendation system, as described in the paper by ...
    """
    def __init__(self, hidden_dim):
        super(MTRec, self).__init__()

        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.q = nn.Parameter(torch.randn(hidden_dim))
        #self.transformer_hist_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8)
        #self.transformer_hist = nn.TransformerEncoder(self.transformer_hist_layer, num_layers=2)
        #self.W_cand = nn.Linear(hidden_dim, hidden_dim)
        #self.W_cand2 = nn.Linear(hidden_dim, hidden_dim)
        #self.dropout = nn.Dropout(0.1)
        #self.layer_norm = nn.LayerNorm(hidden_dim)
        #self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, history, candidates):
        '''
            B - batch size (keep in mind we use an unusual mini-batch approach)
            H - history size (number of articles in the history, usually 30)
            D - hidden size (768)
            history:    B x H x D 
            candidates: B x 1 x D
        '''

        # print(f"{candidates.shape=}")
        #history = self.transformer_hist(history)
        att = self.q * F.tanh(self.W(history))
        att_weight = F.softmax(att, dim=1)
        # print(f"{att_weight.shape=}")

        user_embedding = torch.sum(history * att_weight, dim = 1)
        # print(f"{user_embedding.shape=}")
        # print(f"{user_embedding.unsqueeze(-1).shape=}")
        #candidates = self.norm2(candidates + self.W_cand2(self.layer_norm(self.dropout(F.relu(self.W_cand(candidates))))))

        score = torch.bmm(candidates, user_embedding.unsqueeze(-1)) # B x M x 1
        # print(score.shape)
        return score.squeeze(-1)

    def reshape(self, batch_news, bz):
        n_news = len(batch_news) // bz
        reshaped_batch = batch_news.reshape(bz, n_news, -1)
        return reshaped_batch