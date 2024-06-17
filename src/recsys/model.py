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
torch.set_float32_matmul_precision("medium")


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
    # split_logits = torch.split(logits, repeats.tolist())
    # split_targets = torch.split(one_hot_targets, repeats.tolist())

    # Determine the maximum length for padding
    # max_len = max(repeats)

    # Pad logits and one-hot targets, then stack them
    # padded_logits = torch.stack([F.pad(segment, (0, max_len - segment.size(0)), 'constant', float('-inf')) for segment in split_logits])
    # padded_targets = torch.stack([F.pad(segment, (0, max_len - segment.size(0)), 'constant', 0) for segment in split_targets])

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


class UserEncoder(nn.Module):
    """
    A simple user encoder that averages the embeddings of the user's read articles.
    """

    def __init__(self, hidden_dim):
        super(UserEncoder, self).__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.q = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, history):
        """
        B - batch size (keep in mind we use an unusual mini-batch approach)
        H - history size (number of articles in the history, usually 30)
        D - hidden size (768)
        history: B x H x D
        """
        att = self.q * F.tanh(self.W(history))
        att_weight = F.softmax(att, dim=1)
        user_embedding = torch.sum(history * att_weight, dim=1)
        return user_embedding


class MTRec(nn.Module):
    """The main prediction model for the multi-task recommendation system, as described in the paper by ..."""

    def __init__(self, hidden_dim):
        super(MTRec, self).__init__()

        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.q = nn.Parameter(torch.randn(hidden_dim))
        # self.transformer_hist_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8)
        # self.transformer_hist = nn.TransformerEncoder(self.transformer_hist_layer, num_layers=2)
        # self.W_cand = nn.Linear(hidden_dim, hidden_dim)
        # self.W_cand2 = nn.Linear(hidden_dim, hidden_dim)
        # self.dropout = nn.Dropout(0.1)
        # self.layer_norm = nn.LayerNorm(hidden_dim)
        # self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, history, candidates):
        """
        B - batch size (keep in mind we use an unusual mini-batch approach)
        H - history size (number of articles in the history, usually 30)
        D - hidden size (768)
        history:    B x H x D
        candidates: B x 1 x D
        """

        # print(f"{candidates.shape=}")
        # history = self.transformer_hist(history)
        att = self.q * F.tanh(self.W(history))
        att_weight = F.softmax(att, dim=1)
        # print(f"{att_weight.shape=}")

        user_embedding = torch.sum(history * att_weight, dim=1)
        # print(f"{user_embedding.shape=}")
        # print(f"{user_embedding.unsqueeze(-1).shape=}")
        # candidates = self.norm2(candidates + self.W_cand2(self.layer_norm(self.dropout(F.relu(self.W_cand(candidates))))))

        score = torch.bmm(candidates, user_embedding.unsqueeze(-1)) / torch.sqrt(
            candidates.size(-1)
        )  # B x M x 1
        # print(score.shape)
        return score.squeeze(-1)

    def reshape(self, batch_news, bz):
        n_news = len(batch_news) // bz
        reshaped_batch = batch_news.reshape(bz, n_news, -1)
        return reshaped_batch


class NLLLoss(nn.Module):
    def forward(self, preds, target):
        preds = preds.sigmoid()
        return -torch.log(
            preds.where(target == 1, -torch.inf).exp().sum(dim=-1, keepdims=True)
            / (preds.exp().sum(dim=-1, keepdims=True))
        ).mean()


class CategoryEncoder(nn.Module):
    def __init__(self, hidden_dim, n_categories=5):
        super(CategoryEncoder, self).__init__()
        self.linear = nn.Linear(hidden_dim, n_categories)

    def forward(self, history):
        """
        B - batch size (keep in mind we use an unusual mini-batch approach)
        H - history size (number of articles in the history, usually 30)
        D - hidden size (768)
        history: B x H x D
        """
        return self.linear(history)


class MultitaskRecommender(LightningModule):
    """
    The main prediction model for the multi-task recommendation system we implement.
    """

    def __init__(
        self,
        hidden_dim,
        nhead=8,
        num_layers=4,
        n_categories=5,
        lr=1e-2,
        wd=0.0,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        transformer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, batch_first=True
        )

        self.user_encoder = UserEncoder(hidden_dim)
        self.transformer = nn.TransformerEncoder(transformer, num_layers=num_layers)
        self.category_encoder = CategoryEncoder(hidden_dim, n_categories=n_categories)

        self.predictions = []
        self.labels = []
        self.metric_evaluator = MetricEvaluator(
            self.labels,
            self.predictions,
            metric_functions=[
                AucScore(),
                MrrScore(),
                NdcgScore(k=10),
                NdcgScore(k=5),
                LogLossScore(),
                RootMeanSquaredError(),
                F1Score(),
            ],
        )

        # from torchmetrics.retrieval import RetrievalAUROC
        from torchmetrics.classification import MultilabelAccuracy
        from torchmetrics.classification import MultilabelAUROC

        self.accuracy = MultilabelAccuracy(num_labels=5)
        self.auroc = MultilabelAUROC(num_labels=5)

        # NOTE: Positives are weighted 4 times more than negatives as the dataset is imbalanced.
        # See: https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
        # Would be good if we can find a rationale for this in the literature.
        # self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.ones(1) * 4)

        # # Question: Will it understand this dimensionality? A: Probably yes.
        # [0, 0, 0, 0, 1]
        # [0, 0, 0, 0, 1]
        # [0, 0, 0, 0, 1]
        # [0, 0, 0, 0, 1]
        # [0, 0, 0, 0, 1]
        # [0, 0, 0, 0, 1]

        # [0.1, 0.2, 0.3, 0.4, 0.5]
        # [0.1, 0.2, 0.3, 0.4, 0.5]
        # [0.1, 0.2, 0.3, 0.4, 0.5]
        # [0.1, 0.2, 0.3, 0.4, 0.5]
        # [0.1, 0.2, 0.3, 0.4, 0.5]

        # self.criterion = NLLLoss()
        self.criterion = nn.CrossEntropyLoss()
        self.category_loss = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        print(f"Learning rate: {self.hparams.lr}")
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

    def forward(self, history, candidates):
        """
        LEGEND:
        B - batch size (keep in mind we use an unusual mini-batch approach)
        H - history size (number of articles in the history, usually 30)
        C - candidate length
        D - hidden size (768)
        history:    B x H x D
        candidates: B x C x D

        Returns:
        B x C scores
        """
        # Implement a baseline: LinearRegression, SVM?
        # Suggestion: Concatenate both vectors and pass them through a linear layer? (Only if we have time)
        # Maybe integrate our own BERT and finetune it?
        # history = self.transformer(history)
        # user_embedding = self.transformer(history)
        # user_embedding = torch.sum(history * user_embedding.softmax(dim=1), dim=1)
        # user_embedding = (user_embedding.softmax(dim=1) * user_embedding).sum(dim=1)
        user_embedding = self.user_encoder(history)
        # Normalization in order to reduce the variance of the dot product
        scores = torch.bmm(
            F.normalize(candidates), F.normalize(user_embedding.unsqueeze(-1))
        )
        return scores.squeeze(-1)

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        self.param = next(iter(self.parameters())).detach().clone()

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        self.param = self.param - next(iter(self.parameters())).detach().clone()
        self.log("param_change", self.param.abs().mean(), prog_bar=True)

    def training_step(self, batch, batch_idx):
        history, candidates, category, labels = batch
        scores = self(history, candidates)

        # News Ranking Loss
        news_ranking_loss = self.criterion(scores, labels)
        category_scores = self.category_encoder(history)

        category = torch.nn.functional.one_hot(
            category, num_classes=self.hparams.n_categories
        ).float()
        category_loss = self.category_loss(category_scores, category)

        loss = news_ranking_loss + category_loss
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/news_ranking_loss", news_ranking_loss, prog_bar=True)
        self.log("train/category_loss", category_loss, prog_bar=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()

        self.predictions.clear()
        self.labels.clear()

    def validation_step(self, batch, batch_idx):
        history, candidates, category, labels = batch
        scores = self(history, candidates)

        loss = self.criterion(scores, labels)

        accuracy = self.accuracy(scores, labels)
        auroc = self.auroc(scores, labels.long())
        self.log("validation/accuracy", accuracy, prog_bar=True)
        self.log("validation/auroc", auroc, prog_bar=True)
        self.log("validation/loss", loss, prog_bar=True)
        self.predictions.append(scores.detach().cpu().flatten().float().numpy())
        self.labels.append(labels.detach().cpu().flatten().float().numpy())

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        metrics = self.metric_evaluator.evaluate()
        self.log_dict({f"validation/{k}":v for k, v in metrics.evaluations.items()})

    def test_step(self, batch, batch_idx):
        history, candidates, category, _ = batch
        res = []
        for inview in candidates:
            scores = self(history, inview.unsqueeze(0))
            indices = torch.argsort(scores, descending=True)
            res.append(indices.tolist())

        return res
