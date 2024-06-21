import torch
from ebrec.evaluation.metrics_protocols import (
    AccuracyScore,
    AucScore,
    F1Score,
    LogLossScore,
    MetricEvaluator,
    MrrScore,
    NdcgScore,
    RootMeanSquaredError,
)

from torchmetrics import Accuracy, AUROC
from transformers import AutoTokenizer, BertModel
from sklearn.metrics import roc_auc_score
from pytorch_lightning import LightningModule
from recsys.utils.gradient_surgery import PCGrad
from torch import nn
from torch.nn import functional as F

# Setting to get more matmul performance on Tensor Core capable machines.
torch.set_float32_matmul_precision("medium")


class UserEncoder(nn.Module):
    """
    A simple user encoder that averages the attention of the embeddings of the user's read articles.
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


class NLLLoss(nn.Module):
    def forward(self, preds, target):
        # preds = preds.sigmoid()
        # print(target)
        # print(preds.where(target == 1, -torch.inf))
        # exit()
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


class BERTMultitaskRecommender(LightningModule):
    """
    The main prediction model for the multi-task recommendation system with BERT fine-tuning.
    """

    def __init__(self, epochs=10, lr=1e-3, wd=0.0, steps_per_epoch=None, **kwargs):
        super().__init__()
        self.automatic_optimization = kwargs.get("use_gradient_surgery", False)

        self.save_hyperparameters()
        self.predictions = []
        self.labels = []
        self.bert = BertModel.from_pretrained(
            "google-bert/bert-base-multilingual-cased"
        )
        # self.head = nn.Linear(self.bert.config.hidden_size, num_classes+3+category_num_cls) # 3 for ner
        self.user_encoder = UserEncoder(self.bert.config.hidden_size)

        self.metric_evaluator = MetricEvaluator(
            self.labels,
            self.predictions,
            metric_functions=[
                #AucScore(),
                MrrScore(),
                NdcgScore(k=10),
                NdcgScore(k=5),
                #LogLossScore(),
                #RootMeanSquaredError(),
                #F1Score(),
            ],
        )
        self.indx = 0

        self.accuracy = Accuracy(task="multilabel", num_labels=5)
        self.auc_roc = AUROC(task="multiclass", num_classes=5)
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

        # # Question: Is pos weight correct? TODO: Experiment with both.
        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = NLLLoss()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd
        )

        if self.hparams.use_gradient_surgery:
            optimizer = PCGrad(optimizer)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            pct_start=0.1,
            steps_per_epoch=self.hparams.steps_per_epoch,
            epochs=self.hparams.epochs,
            anneal_strategy="linear",
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "frequency": 1,
                "interval": "step",
            },
        }

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
        # history = self.transformer(history)
        # user_embedding = self.transformer(history).mean(dim=1)
        self.indx += 1
        batch_size, hist_size, seq_len = history["input_ids"].size()
        history["input_ids"] = history["input_ids"].view(
            batch_size * hist_size, seq_len
        )
        history["attention_mask"] = history["attention_mask"].view(
            batch_size * hist_size, seq_len
        )
        history["token_type_ids"] = history["token_type_ids"].view(
            batch_size * hist_size, seq_len
        )
        history = (
            self.bert(**history)
            .last_hidden_state[:, 0, :]
            .view(batch_size, hist_size, -1)
        )

        batch_size, cand_size, seq_len = candidates["input_ids"].size()
        candidates["input_ids"] = candidates["input_ids"].view(
            batch_size * cand_size, seq_len
        )
        candidates["attention_mask"] = candidates["attention_mask"].view(
            batch_size * cand_size, seq_len
        )
        candidates["token_type_ids"] = candidates["token_type_ids"].view(
            batch_size * cand_size, seq_len
        )
        candidates = (
            self.bert(**candidates)
            .last_hidden_state[:, 0, :]
            .view(batch_size, cand_size, -1)
        )

        user_embedding = self.user_encoder(history)
        if self.indx % 100 == 0:
            print(candidates)
        # Normalization in order to reduce the variance of the dot product
        scores = torch.bmm(
            candidates, user_embedding.unsqueeze(-1)
            # F.normalize(candidates, dim=-1),
            # F.normalize(user_embedding.unsqueeze(-1), dim=1),
        )
        return scores.squeeze(-1)

    def compute_loss(self, batch):
        # News Ranking Loss
        history, candidates, labels = batch

        categories = history.pop("category")
        _ = candidates.pop("category")
        scores = self(history, candidates)
        news_ranking_loss = self.criterion(scores, labels)

        # category_scores = self.category_encoder(history)
        # category = torch.nn.functional.one_hot(
        # category, num_classes=self.hparams.n_categories
        # ).float()
        # category_loss = self.category_loss(category_scores, category)

        category_loss = torch.tensor(0.0, device=self.device)
        return {
            "news_ranking_loss": news_ranking_loss,
            "category_loss": category_loss,
            "scores": scores,
            "labels": labels,
        }

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)

        news_ranking_loss = loss["news_ranking_loss"]
        category_loss = loss["category_loss"]
        #
        aux_loss = 0.3 * category_loss
        loss = news_ranking_loss
        # Gradient Surgery
        # ================
        if self.hparams.use_gradient_surgery:
            optimizer = self.optimizers()
            optimizer.zero_grad()

            optimizer.optimizer.pc_backward([news_ranking_loss, aux_loss])
            optimizer.step()
        # ================

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/news_ranking_loss", news_ranking_loss)
        self.log("train/category_loss", category_loss)
        return loss

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()

        self.predictions.clear()
        self.labels.clear()

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)

        scores = loss["scores"]
        labels = loss["labels"]
        # labels_indices = labels.argmax(dim=-1)

        # accuracy = self.accuracy(scores, labels_indices)
        # auc_roc = self.auc_roc(scores, labels_indices)
        # self.log("validation/accuracy", accuracy)
        # self.log("validation/auc_roc", auc_roc)
        self.log("validation/loss", loss["news_ranking_loss"], prog_bar=True)
        self.predictions.append(scores.detach().cpu().flatten().float().numpy())
        self.labels.append(labels.detach().cpu().flatten().float().numpy())

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        metrics = self.metric_evaluator.evaluate()
        self.log_dict({f"validation/{k}": v for k, v in metrics.evaluations.items()})

    def test_step(self, batch, batch_idx):
        res = []
        
        histories, candidates = batch
        for hist, cand in zip(histories, candidates):
            scores = self(hist, cand)
            indices = torch.argsort(scores, descending=True) + 1
            res.append((scores.tolist(), indices.tolist()))

        return res


class MultitaskRecommender(BERTMultitaskRecommender):
    """
    The main prediction model for the multi-task recommendation system we implement.
    """

    def __init__(
        self,
        hidden_dim,
        nhead=8,
        num_layers=2,
        n_categories=5,
        lr=1e-2,
        wd=0.0,
        use_gradient_surgery=False,
        **kwargs,
    ):
        super().__init__()
        self.automatic_optimization = use_gradient_surgery

        self.save_hyperparameters()

        # transformer = nn.TransformerEncoderLayer(
        #     d_model=hidden_dim, nhead=nhead, batch_first=True
        # )
        # self.transformer = nn.TransformerEncoder(transformer, num_layers=num_layers)

        self.user_encoder = UserEncoder(hidden_dim)
        self.category_encoder = CategoryEncoder(hidden_dim, n_categories=n_categories)

        del self.bert

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

        # history = self.transformer(history)
        # user_embedding = self.transformer(history)
        # user_embedding = torch.sum(history * user_embedding.softmax(dim=1), dim=1)
        # user_embedding = (user_embedding.softmax(dim=1) * user_embedding).sum(dim=1)
        user_embedding = self.user_encoder(history)

        # Normalization in order to reduce the variance of the dot product
        scores = torch.bmm(
            F.normalize(candidates, dim=-1),
            F.normalize(user_embedding.unsqueeze(-1), dim=1),
        )

        scores = scores.squeeze(-1)
        return scores

    def compute_loss(self, batch):
        # News Ranking Loss
        history, candidates, labels = batch

        scores = self(history, candidates)
        news_ranking_loss = self.criterion(scores, labels)

        # category_scores = self.category_encoder(history)
        # category = torch.nn.functional.one_hot(
        # category, num_classes=self.hparams.n_categories
        # ).float()
        # category_loss = self.category_loss(category_scores, category)

        category_loss = torch.tensor(0.0, device=self.device)
        return {
            "news_ranking_loss": news_ranking_loss,
            "category_loss": category_loss,
            "scores": scores,
            "labels": labels,
        }

'''
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

    loss = self.criterion(scores, labels)
    #if self.indx % 100 == 0:
    #    print(scores)
    #    print(labels)
    accuracy = self.accuracy(scores.float(), labels.argmax(dim=-1))
    self.log("validation/accuracy", accuracy)
    self.log("validation/loss", loss, prog_bar=True)
    self.predictions.append(scores.detach().cpu().flatten().float().numpy())
    self.labels.append(labels.detach().cpu().flatten().float().numpy())

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
'''
