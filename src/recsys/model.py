import torch
from torch import nn
from torch.nn import functional as F

import torch.nn.functional as F

def apply_softmax_crossentropy(logits, repeats, one_hot_targets):
    """
    Applies softmax and computes the cross-entropy loss for each segment of logits with one-hot encoded targets.

    Args:
        logits (torch.Tensor): Flattened array of logits.
        repeats (torch.Tensor): Tensor indicating the number of logits in each segment.
        one_hot_targets (torch.Tensor): Flattened one-hot encoded target labels.

    Returns:
        torch.Tensor: The cross-entropy loss for each segment.
    """
    assert logits.ndim == 1, "Logits should be a flattened array"
    assert one_hot_targets.ndim == 1, "One-hot targets should be a flattened array"

    # Split logits and one-hot targets according to repeats
    split_logits = torch.split(logits, repeats.tolist())
    split_targets = torch.split(one_hot_targets, repeats.tolist())

    # Determine the maximum length for padding
    max_len = max(repeats)
    
    # Pad logits and one-hot targets, then stack them
    padded_logits = torch.stack([F.pad(segment, (0, max_len - len(segment)), 'constant', float('-inf')) for segment in split_logits])
    padded_targets = torch.stack([F.pad(segment, (0, max_len - len(segment)), 'constant', 0) for segment in split_targets])
    
    # Reshape the padded targets to their original shape
    num_classes = int(one_hot_targets.size(0) / repeats.sum())
    reshaped_targets = padded_targets.view(-1, num_classes)
    
    # Apply softmax to logits
    softmaxed_logits = F.log_softmax(padded_logits.view(-1, num_classes), dim=-1)
    
    # Calculate cross-entropy loss
    losses = -torch.sum(reshaped_targets * softmaxed_logits, dim=-1)
    
    # Reshape the losses back to the original segments
    losses = losses.view(len(repeats), max_len)
    
    # Mask out the padded positions
    mask = (reshaped_targets.sum(dim=-1) > 0).float().view(len(repeats), max_len)
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