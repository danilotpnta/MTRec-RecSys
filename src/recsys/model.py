from pytorch_lightning import LightningModule
import torch
from torch import nn
from torch.nn import functional as F
from ebrec.evaluation.metrics_protocols import MetricEvaluator
from ebrec.evaluation.metrics_protocols import (
    AucScore,
    MrrScore,
    NdcgScore,
    LogLossScore,
    RootMeanSquaredError,
    AccuracyScore,
    F1Score,
)

# Setting to get more matmul performance on Tensor Core capable machines.
torch.set_float32_matmul_precision('medium')


def apply_softmax_crossentropy(logits, repeats, one_hot_targets, epsilon=1e-10):
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
    assert logits.ndim == 1, "Logits should be a flattened array"
    assert one_hot_targets.ndim == 1, "One-hot targets should be a flattened array"

    # Split logits and one-hot targets according to repeats
    split_logits = torch.split(logits, repeats.tolist())
    split_targets = torch.split(one_hot_targets, repeats.tolist())

    # Determine the maximum length for padding
    max_len = max(repeats)

    # Pad logits and one-hot targets, then stack them
    padded_logits = torch.stack(
        [
            F.pad(segment, (0, max_len - segment.size(0)), "constant", float("-inf"))
            for segment in split_logits
        ]
    )
    padded_targets = torch.stack(
        [
            F.pad(segment, (0, max_len - segment.size(0)), "constant", 0)
            for segment in split_targets
        ]
    )

    # Apply softmax to logits and add epsilon to avoid NaNs
    softmaxed_logits = F.softmax(padded_logits, dim=-1)
    log_softmaxed_logits = torch.log(softmaxed_logits + epsilon)

    # Calculate cross-entropy loss
    losses = -torch.sum(padded_targets * log_softmaxed_logits, dim=-1)

    # Mask out the padded positions
    mask = (padded_targets.sum(dim=-1) > 0).float()
    masked_losses = losses * mask

    # Sum the losses for each segment and divide by the number of true (non-padded) entries
    segment_losses = masked_losses.sum(dim=-1) / mask.sum(dim=-1)

    return segment_losses


class MTRec(nn.Module):
    """The main prediction model for the multi-task recommendation system, as described in the paper by ..."""

    def __init__(self, hidden_dim):
        super(MTRec, self).__init__()

        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.q = nn.Parameter(torch.randn(hidden_dim))
        self.transformer_hist_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8
        )
        self.transformer_cand_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8
        )
        self.transformer_hist = nn.TransformerEncoder(
            self.transformer_hist_layer, num_layers=2
        )
        self.transformer_cand = nn.TransformerEncoder(
            self.transformer_cand_layer, num_layers=2
        )

    def forward(self, history, candidates):
        """
        B - batch size (keep in mind we use an unusual mini-batch approach)
        H - history size (number of articles in the history, usually 30)
        D - hidden size (768)
        history:    B x H x D
        candidates: B x 1 x D
        """

        # print(f"{candidates.shape=}")
        history = self.transformer_hist(history)
        att = self.q * F.tanh(self.W(history))
        att_weight = F.softmax(att, dim=1)
        # print(f"{att_weight.shape=}")

        user_embedding = torch.sum(history * att_weight, dim=1)
        # print(f"{user_embedding.shape=}")
        # print(f"{user_embedding.unsqueeze(-1).shape=}")
        candidates = self.transformer_cand(candidates)
        score = torch.bmm(candidates, user_embedding.unsqueeze(-1))  # B x M x 1
        # print(score.shape)
        return score.squeeze(-1)

    def reshape(self, batch_news, bz):
        n_news = len(batch_news) // bz
        reshaped_batch = batch_news.reshape(bz, n_news, -1)
        return reshaped_batch


class MultitaskRecommender(LightningModule):
    """
    The main prediction model for the multi-task recommendation system we implement.
    """

    def __init__(self, hidden_dim, nhead=8, num_layers=4, lr=1e-3, wd=0.0, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        transformer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer, num_layers=num_layers)

        self.predictions = []
        self.labels = []
        self.metric_evaluator = MetricEvaluator(
            self.labels,
            self.predictions,
            metric_functions=[
                AucScore(),
                MrrScore(),
                NdcgScore(k=10),
                LogLossScore(),
                RootMeanSquaredError(),
                AccuracyScore(),
                F1Score(),
            ],
        )

        # NOTE: Positives are weighted 4 times more than negatives as the dataset is imbalanced.
        # See: https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
        # Would be good if we can find a rationale for this in the literature.
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.ones(1) * 4)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd
        )

    def forward(self, history, candidates):
        """
        LEGEND:
        B - batch size (keep in mind we use an unusual mini-batch approach)
        H - history size (number of articles in the history, usually 30)
        D - hidden size (768)
        history:    B x H x D
        candidates: B x 1 x D
        """
        user_embedding = self.transformer(history).mean(1)
        # Normalization in order to reduce the variance of the dot product
        scores = torch.bmm(
            F.normalize(candidates), F.normalize(user_embedding.unsqueeze(-1))
        )
        return scores.squeeze(-1)

    def training_step(self, batch, batch_idx):
        history, candidates, labels = batch
        scores = self(history, candidates)

        loss = self.criterion(scores, labels)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()

        self.predictions.clear()
        self.labels.clear()

    def validation_step(self, batch, batch_idx):
        history, candidates, labels = batch
        scores = self(history, candidates)

        loss = self.criterion(scores, labels)

        self.log("val_loss", loss, prog_bar=True)
        self.predictions.append(scores.detach().cpu().flatten().float().numpy())
        self.labels.append(labels.detach().cpu().flatten().float().numpy())

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        metrics = self.metric_evaluator.evaluate()
        self.log_dict(metrics.evaluations)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
