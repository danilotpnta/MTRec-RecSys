import polars as pl

from torch.utils.data import Dataset
from recsys.dataset import sampling_strategy

import torch

HISTORY_SIZE = 16
COL_IMPRESSION_ID = "Impression ID"  # The ID of an impression.
COL_USER_ID = "User ID"  # The anonymous ID of a user.
COL_TIME = "Time"  # The impression time with format "MM/DD/YYYY HH:MM:SS AM/PM".
COL_HISTORY = "History"  # The news click history (ID list of clicked news) of this user before this impression. The clicked news articles are ordered by time.
COL_IMPRESSIONS = "Impressions"  # List of news displayed in this impression and user's click behaviors on them (1 for click and 0 for non-click). The orders of news in a impressions have been shuffled.

COL_NEWS_ID = "News ID"
COL_CATEGORY = "Category"
COL_SUBCATEGORY = "SubCategory"
COL_TITLE = "Title"
COL_ABSTRACT = "Abstract"
COL_URL = "URL"
COL_TITLE_ENTITIES = "Title Entities"
COL_ABSTRACT_ENTITIES = "Abstract Entities"
COL_CATEGORY_ID = "Category ID"


class MINDDataset(Dataset):
    def __init__(
        self,
        behaviors,
        articles,
        tokenizer=None,
        max_length=64,
        history_size=HISTORY_SIZE,
        max_labels=5,
    ):
        self.behaviors = behaviors
        self.articles = articles

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.history_size = history_size
        self.max_labels = max_labels

        self.articles, self.article2id, self.id2article = self._process_articles(
            articles
        )

        self.behaviors = self._process_behaviors(
            behaviors, self.article2id, history_size
        )

    @classmethod
    def _process_behaviors(cls, behaviors, article2id, history_size=HISTORY_SIZE):
        # Preprocess the behaviors
        behaviors = behaviors.with_columns(
            pl.col(COL_HISTORY).str.split(" "),
            pl.col(COL_IMPRESSIONS).str.split(" "),
            pl.col(COL_TIME).str.to_datetime("%m/%d/%Y %I:%M:%S %p"),
        )
        behaviors = behaviors.fill_null(0)

        # Extract the labels and impressions
        grouped = behaviors[COL_IMPRESSIONS].list.eval(pl.element().str.split("-"))
        labels = grouped.list.eval(pl.element().list.last().cast(pl.UInt8)).alias(
            "labels"
        )
        impressions = grouped.list.eval(pl.element().list.first())

        behaviors = behaviors.with_columns(impressions, labels)

        # Replace the article IDs with the new IDs
        for col in [COL_HISTORY, COL_IMPRESSIONS]:
            behaviors = behaviors.with_columns(
                replace_column_from_mapping(behaviors[col], col, article2id)
            )

        # Pad the history
        return behaviors.with_columns(
            pl.col(COL_HISTORY)
            .list.reverse()
            .list.eval(pl.element().extend_constant(0, history_size))
            .list.reverse()
            .list.tail(history_size)
        )

    def _process_articles(self, articles: pl.DataFrame, col_news_id=COL_NEWS_ID):
        articles = articles.with_columns(
            pl.col(COL_CATEGORY)
            .cast(pl.Categorical("lexical"))
            .to_physical()
            .cast(pl.UInt8)
            .alias(COL_CATEGORY_ID),
        )

        lookup_matrix = [""] + articles[COL_TITLE].to_list()
        self.lookup_matrix = self.tokenizer(
            lookup_matrix,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        self.lookup_matrix["input_ids"][0][0] = 0
        self.lookup_matrix["attention_mask"][0][0] = 0
        self.lookup_matrix["input_ids"][0][1] = 0
        self.lookup_matrix["attention_mask"][0][1] = 0
        self.lookup_matrix["category"] = torch.tensor(
            [0] + articles[COL_CATEGORY_ID].to_list()
        )
        self.lookup_matrix["sentiment_label"] = torch.tensor([0] * (1+len(articles)))

        self.max_categories = self.lookup_matrix["category"].max().item() + 1
        self.max_sentiment_labels = self.lookup_matrix["sentiment_label"].max().item() + 1

        id2article = {i: id for i, id in enumerate(articles[col_news_id], start=1)}
        article2id = {id: i for i, id in id2article.items()}

        return articles, article2id, id2article

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, index: int):
        _hist = self.behaviors[COL_HISTORY][index].to_list()

        history = {
            key: torch.stack([self.lookup_matrix[key][_] for _ in _hist])
            for key in self.lookup_matrix.data.keys()
        }

        _cand = self.behaviors[COL_IMPRESSIONS][index].to_numpy()
        labels = self.behaviors["labels"][index].to_numpy()
        idxs = sampling_strategy(labels, self.max_labels)
        _cand = _cand[idxs]
        candidates = {
            key: torch.stack([self.lookup_matrix[key][_] for _ in _cand])
            for key in self.lookup_matrix.data.keys()
        }

        labels = torch.tensor(labels[idxs], dtype=torch.float32)

        return history, candidates, labels


def replace_column_from_mapping(
    df: pl.Series, column: str, mapping: dict, default_value=0, return_dtype=pl.Int64
):
    """
    Replace values inside a list column with a mapping dictionary.

    Converts the column to a DataFrame, adds a grouping column, explodes the dataframe, replaces the values, and then groups by the index to restore the original shape.

    Seems to be much faster than using eval expressions.
    """
    return (
        df.to_frame()
        .with_row_index("__index")
        .explode(column)
        .with_columns(
            pl.col(column).replace(
                mapping, return_dtype=return_dtype, default=default_value
            )
        )
        .group_by("__index")
        .agg(pl.col(column))
        .get_column(column)
    )
