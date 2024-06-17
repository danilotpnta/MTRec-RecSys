import os.path
from random import shuffle
from typing import Any, Literal
from recsys.utils.classes import PolarsDataFrameWrapper
from recsys.utils.download import CHALLENGE_DATASET, download_file, unzip_file

import numpy as np
import polars as pl
import torch
from ebrec.utils._behaviors import create_binary_labels_column, truncate_history
from ebrec.utils._constants import (
    DEFAULT_ARTICLE_ID_COL,
    DEFAULT_BODY_COL,
    DEFAULT_CATEGORY_STR_COL,
    DEFAULT_CATEGORY_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_IMPRESSION_ID_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_LABELS_COL,
    DEFAULT_SUBTITLE_COL,
    DEFAULT_TITLE_COL,
    DEFAULT_TOPICS_COL,
    DEFAULT_USER_COL,
)
from ebrec.utils._polars import slice_join_dataframes
from ebrec.utils._python import (
    create_lookup_dict,
    create_lookup_objects,
    generate_unique_name,
)
from torch.utils.data import Dataset
from pytorch_lightning import LightningDataModule
import json

COLUMNS = [
    DEFAULT_USER_COL,
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_IMPRESSION_ID_COL,
]


DEFAULT_TOKENS_COL = "tokens"
N_SAMPLES_COL = "n_samples"


class NewsDataset(Dataset):
    behaviors: pl.DataFrame
    history: pl.DataFrame
    articles: pl.DataFrame

    def __init__(
        self,
        tokenizer,
        behaviors: pl.DataFrame,
        history: pl.DataFrame,
        articles: pl.DataFrame,
        history_size: int = 30,
        max_labels: int = 5,
        padding_value: int = 0,
        max_length=128,
        embeddings_path=None,
        neg_count=5,
    ):
        self.behaviors = behaviors
        self.history = history
        self.articles = articles
        self.history_size = history_size
        self.padding_value = padding_value
        self.neg_count = neg_count

        # self.tokenizer = tokenizer
        # self.max_length = max_length
        self.max_labels = max_labels

        # NOTE: Keep an eye on this if memory issues arise
        self.articles = self.articles.select(
            [
                DEFAULT_ARTICLE_ID_COL,  # article_id
                DEFAULT_TITLE_COL,  # title
                DEFAULT_BODY_COL,  # body
                DEFAULT_SUBTITLE_COL,  # subtitle
                DEFAULT_TOPICS_COL,  # topics
                DEFAULT_CATEGORY_STR_COL,  # category_str
            ]
        )

        self.history = self._process_history(self.history, history_size, padding_value)

        # TODO (Matey): Decided to instead only use pre-computed embeddings for now. You might want to look into this later down the line and implement custom embeddings (and e.g. train BERT as well).
        self._prepare_training_data(embeddings_path)

    # @staticmethod
    # def from_preprocessed(path):

    def save_preprocessed(self, path: str):
        """Save the preprocessed data to the given path directory."""
        data = {
            "history_size": self.history_size,
            "padding_value": self.padding_value,
            "max_labels": self.max_labels,
            "max_categories": self.max_categories
        }

        np.save(path + "/lookup_matrix.npy", self.lookup_matrix)
        np.save(path + "/aux_cat_lookup_matrix.npy", self.aux_cat_lookup_matrix)
        with open(path + "/parameters.json", "w") as f:
            json.dump(data, f)
        self.behaviors.write_parquet(path + "/behaviors.parquet")
        self.history.write_parquet(path + "/history.parquet")
        self.articles.write_parquet(path + "/articles.parquet")
        self.data.dataframe.write_parquet(path + "/data.parquet")

    @staticmethod
    def from_preprocessed(path: str):
        """Load the preprocessed data from the given path directory."""
        dataset = NewsDataset.__new__(NewsDataset)
        with open(path + "/parameters.json", "r") as f:
            data = json.load(f)
            dataset.history_size = data["history_size"]
            dataset.padding_value = data["padding_value"]
            dataset.max_labels = data["max_labels"]
            dataset.max_categories = data["max_categories"]

        dataset.lookup_matrix = np.load(path + "/lookup_matrix.npy")
        dataset.aux_cat_lookup_matrix = np.load(path + "/aux_cat_lookup_matrix.npy")
        dataset.behaviors = pl.read_parquet(path + "/behaviors.parquet")
        dataset.history = pl.read_parquet(path + "/history.parquet")
        dataset.articles = pl.read_parquet(path + "/articles.parquet")
        dataset.data = PolarsDataFrameWrapper(pl.read_parquet(path + "/data.parquet"))

        return dataset

    @classmethod
    def _process_history(
        cls, history: pl.LazyFrame, history_size: int = 30, padding_value: int = 0
    ) -> pl.DataFrame:
        return (
            history.select(
                [
                    DEFAULT_USER_COL,  # "user_id"
                    DEFAULT_HISTORY_ARTICLE_ID_COL,  # article_id_fixed
                ]
            )
            .pipe(
                truncate_history,
                column=DEFAULT_HISTORY_ARTICLE_ID_COL,
                history_size=history_size,
                padding_value=padding_value,
                enable_warning=False,
            )
            .collect()
        )

    def _prepare_training_data(self, embeddings_path=None):
        assert (
            embeddings_path is not None
        ), "You need to provide a path to the embeddings file."
        embeddings = pl.read_parquet(embeddings_path)

        self.articles = (
            self.articles.lazy()
            .join(embeddings.lazy(), on=DEFAULT_ARTICLE_ID_COL, how="inner")
            .rename({embeddings.columns[-1]: DEFAULT_TOKENS_COL})
            .cast({DEFAULT_CATEGORY_STR_COL: pl.Categorical})
            .with_columns(
                pl.col(DEFAULT_CATEGORY_STR_COL)
                .to_physical()
                .alias(DEFAULT_CATEGORY_COL)
            )
            .collect()
        )

        # ArticleID2Vec lookup
        article_dict = create_lookup_dict(
            self.articles.select(DEFAULT_ARTICLE_ID_COL, DEFAULT_TOKENS_COL),
            key=DEFAULT_ARTICLE_ID_COL,
            value=DEFAULT_TOKENS_COL,
        )

        # Create the lookup indexes and matrix
        self.lookup_indexes, self.lookup_matrix = create_lookup_objects(
            article_dict, unknown_representation="zeros"
        )

        self.aux_cat_lookup_matrix = (
            self.articles[DEFAULT_CATEGORY_COL].to_numpy().astype(np.uint8)
        )
        self.max_categories = self.aux_cat_lookup_matrix.max().item() + 1

        # Prepare the actual training data
        self.behaviors = self.behaviors.collect()

        self.data = (
            slice_join_dataframes(
                df1=self.behaviors,
                df2=self.history,
                on=DEFAULT_USER_COL,
                how="left",
            )
            .select(COLUMNS)
            .pipe(create_binary_labels_column, seed=42, label_col=DEFAULT_LABELS_COL)
            .pipe(sort_and_select, n=self.max_labels)
            .with_columns(pl.col(DEFAULT_LABELS_COL).list.len().alias(N_SAMPLES_COL))
            .with_columns()
        )

        self.data = (
            self.data.lazy()
            .pipe(
                map_list_article_id_to_value,
                behaviors_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
                mapping=self.lookup_indexes,
                fill_nulls=[0],
            )
            .pipe(
                map_list_article_id_to_value,
                behaviors_column=DEFAULT_INVIEW_ARTICLES_COL,
                mapping=self.lookup_indexes,
                fill_nulls=[0],
            )
            .collect(streaming=True)
        )

        self.data = PolarsDataFrameWrapper(self.data)

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, index):
        """
        Get the samples for the given index.

        Args:
            index (int): An integer or a slice index.

        Returns:
            history: torch.Tensor: The history input features.
            candidate: torch.Tensor: The candidate input features.
            y: torch.Tensor: The target labels.
        """

        batch = self.data[index]
        _hist = batch[DEFAULT_HISTORY_ARTICLE_ID_COL].to_list()

        # Grab history and candidate vectors
        history_input = self.lookup_matrix[_hist]
        candidate_input = self.lookup_matrix[
            batch[DEFAULT_INVIEW_ARTICLES_COL].to_list()
        ]
        # =>
        history_input = torch.from_numpy(history_input).squeeze()
        candidate_input = torch.from_numpy(candidate_input).squeeze()
        y = torch.tensor(batch[DEFAULT_LABELS_COL], dtype=torch.float32).squeeze()

        category = self.aux_cat_lookup_matrix[_hist]

        category = torch.from_numpy(category).long().squeeze()
        # ========================
        return history_input, candidate_input, category, y


class NewsDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        tokenizer: Any = None,
        batch_size: int = 32,
        history_size: int = 30,
        max_labels: int = 5,
        padding_value: int = 0,
        max_length: int = 128,
        num_workers: int = 0,
        dataset: Literal["demo", "small", "large", "test"] = "demo",
        embeddings: Literal[
            "xlm-roberta-base", "bert-base-cased", "word2vec", "contrastive_vector"
        ] = "xlm-roberta-base",
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.root_path = data_path
        self.batch_size = batch_size
        self.history_size = history_size
        self.max_labels = max_labels
        self.padding_value = padding_value
        self.max_length = max_length
        self.num_workers = num_workers

        self.dataset = dataset
        self.embeddings = embeddings

    def prepare_data(self):
        # Download the dataset
        url = CHALLENGE_DATASET[self.dataset]
        savefolder = os.path.join(self.root_path, self.dataset)
        if not os.path.exists(savefolder):
            os.makedirs(savefolder, exist_ok=True)
            filename = download_file(
                url, os.path.join(savefolder, url.rpartition("/")[-1])
            )
            self.data_path = unzip_file(filename, savefolder)
            os.remove(filename)
        else:
            self.data_path = savefolder

        # Download the embeddings
        embeddings_url = CHALLENGE_DATASET[self.embeddings]
        embeddings_folder = os.path.join(self.root_path, self.embeddings)
        if not os.path.exists(embeddings_folder):
            os.makedirs(embeddings_folder, exist_ok=True)
            filename = download_file(
                embeddings_url,
                os.path.join(embeddings_folder, embeddings_url.rpartition("/")[-1]),
            )
            self.embeddings_path = unzip_file(filename, embeddings_folder)
            os.remove(filename)
        else:
            self.embeddings_path = embeddings_folder

        # Find the parquet file
        for root, dirs, files in os.walk(self.embeddings_path):
            for file in files:
                if file.endswith(".parquet"):
                    self.embeddings_path = os.path.join(root, file)
                    return
        raise FileNotFoundError("No parquet file found in the embeddings directory.")

    def _download_test(self):
        # Download the dataset
        url = CHALLENGE_DATASET["test"]
        savefolder = os.path.join(self.root_path, "test")
        if not os.path.exists(savefolder):
            os.makedirs(savefolder, exist_ok=True)
            filename = download_file(
                url, os.path.join(savefolder, url.rpartition("/")[-1])
            )
            self.data_path = unzip_file(filename, savefolder)
            os.remove(filename)
        else:
            self.data_path = savefolder

    def setup(self, stage: str = None):
        match stage:
            case "fit" | "validation" | None:
                # Load the training data
                if not hasattr(self, "train_dataset"):
                    save_dir = os.path.join(self.data_path, "train", "preprocessed")
                    if os.path.exists(save_dir):
                        self.train_dataset = NewsDataset.from_preprocessed(save_dir)

                    else:
                        df_behaviors, df_history, df_articles = load_data(
                            self.data_path, split="train"
                        )
                        self.train_dataset = NewsDataset(
                            tokenizer=self.tokenizer,
                            behaviors=df_behaviors,
                            history=df_history,
                            articles=df_articles,
                            history_size=self.history_size,
                            max_labels=self.max_labels,
                            padding_value=self.padding_value,
                            max_length=self.max_length,
                            embeddings_path=self.embeddings_path,
                        )
                        os.makedirs(save_dir, exist_ok=True)
                        self.train_dataset.save_preprocessed(save_dir)

                if not hasattr(self, "val_dataset"):
                    # Load the validation data
                    save_dir = os.path.join(self.data_path, "validation", "preprocessed")
                    if os.path.exists(save_dir):
                        self.val_dataset = NewsDataset.from_preprocessed(save_dir)
                    else:
                        df_behaviors, df_history, df_articles = load_data(
                            self.data_path, split="validation"
                        )
                        self.val_dataset = NewsDataset(
                            tokenizer=self.tokenizer,
                            behaviors=df_behaviors,
                            history=df_history,
                            articles=df_articles,
                            history_size=self.history_size,
                            max_labels=self.max_labels,
                            padding_value=self.padding_value,
                            max_length=self.max_length,
                            embeddings_path=self.embeddings_path,
                        )
                        os.makedirs(save_dir, exist_ok=True)
                        self.val_dataset.save_preprocessed(save_dir)

            case _:
                # Otherwise, test.
                if not hasattr(self, "test_dataset"):
                    self._download_test()
                    df_behaviors, df_history, df_articles = load_data(
                        self.data_path, split="test"
                    )
                    self.test_dataset = NewsDataset(
                        tokenizer=self.tokenizer,
                        behaviors=df_behaviors,
                        history=df_history,
                        articles=df_articles,
                        history_size=self.history_size,
                        max_labels=self.max_labels,
                        padding_value=self.padding_value,
                        max_length=self.max_length,
                        embeddings_path=self.embeddings_path,
                    )
                    os.makedirs(save_dir, exist_ok=True)
                    self.test_dataset.save_preprocessed(save_dir)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=bool(self.num_workers),
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=bool(self.num_workers),
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=bool(self.num_workers),
        )


def load_data(
    data_path: str,
    split="train",
):
    """
    Load the data from the given path and return the dataset.

    Parameters
    ----------
    tokenizer : Any
        The tokenizer to use for tokenization.
    data_path : str
        The path to the data.
    split : str, optional
        Split to use (train or validation), by default "train"
    Returns
    -------
    Tuple[LazyFrame, LazyFrame, LazyFrame]
    """
    _data_path = os.path.join(data_path, split)

    df_behaviors = pl.scan_parquet(_data_path + "/behaviors.parquet")
    df_history = pl.scan_parquet(_data_path + "/history.parquet")
    df_articles = pl.scan_parquet(data_path + "/articles.parquet")

    return df_behaviors, df_history, df_articles


def map_list_article_id_to_value(
    behaviors: pl.LazyFrame,
    behaviors_column: str,
    mapping: dict[int, pl.Series],
    drop_nulls: bool = False,
    fill_nulls: any = None,
) -> pl.LazyFrame:
    """

    Maps the values of a column in a DataFrame `behaviors` containing article IDs to their corresponding values
    in a column in another DataFrame `articles`. The mapping is performed using a dictionary constructed from
    the two DataFrames. The resulting DataFrame has the same columns as `behaviors`, but with the article IDs
    replaced by their corresponding values.

    Args:
        behaviors (pl.DataFrame): The DataFrame containing the column to be mapped.
        behaviors_column (str): The name of the column to be mapped in `behaviors`.
        mapping (dict[int, pl.Series]): A dictionary with article IDs as keys and corresponding values as values.
            Note, 'replace' works a lot faster when values are of type pl.Series!
        drop_nulls (bool): If `True`, any rows in the resulting DataFrame with null values will be dropped.
            If `False` and `fill_nulls` is specified, null values in `behaviors_column` will be replaced with `fill_null`.
        fill_nulls (Optional[any]): If specified, any null values in `behaviors_column` will be replaced with this value.

    Returns:
        pl.DataFrame: A new DataFrame with the same columns as `behaviors`, but with the article IDs in
            `behaviors_column` replaced by their corresponding values in `mapping`.

    Example:
    >>> behaviors = pl.DataFrame(
            {"user_id": [1, 2, 3, 4, 5], "article_ids": [["A1", "A2"], ["A2", "A3"], ["A1", "A4"], ["A4", "A4"], None]}
        )
    >>> articles = pl.DataFrame(
            {
                "article_id": ["A1", "A2", "A3"],
                "article_type": ["News", "Sports", "Entertainment"],
            }
        )
    >>> articles_dict = dict(zip(articles["article_id"], articles["article_type"]))
    >>> map_list_article_id_to_value(
            behaviors=behaviors,
            behaviors_column="article_ids",
            mapping=articles_dict,
            fill_nulls="Unknown",
        )
        shape: (4, 2)
        ┌─────────┬─────────────────────────────┐
        │ user_id ┆ article_ids                 │
        │ ---     ┆ ---                         │
        │ i64     ┆ list[str]                   │
        ╞═════════╪═════════════════════════════╡
        │ 1       ┆ ["News", "Sports"]          │
        │ 2       ┆ ["Sports", "Entertainment"] │
        │ 3       ┆ ["News", "Unknown"]         │
        │ 4       ┆ ["Unknown", "Unknown"]      │
        │ 5       ┆ ["Unknown"]                 │
        └─────────┴─────────────────────────────┘
    >>> map_list_article_id_to_value(
            behaviors=behaviors,
            behaviors_column="article_ids",
            mapping=articles_dict,
            drop_nulls=True,
        )
        shape: (4, 2)
        ┌─────────┬─────────────────────────────┐
        │ user_id ┆ article_ids                 │
        │ ---     ┆ ---                         │
        │ i64     ┆ list[str]                   │
        ╞═════════╪═════════════════════════════╡
        │ 1       ┆ ["News", "Sports"]          │
        │ 2       ┆ ["Sports", "Entertainment"] │
        │ 3       ┆ ["News"]                    │
        │ 4       ┆ null                        │
        │ 5       ┆ null                        │
        └─────────┴─────────────────────────────┘
    >>> map_list_article_id_to_value(
            behaviors=behaviors,
            behaviors_column="article_ids",
            mapping=articles_dict,
            drop_nulls=False,
        )
        shape: (4, 2)
        ┌─────────┬─────────────────────────────┐
        │ user_id ┆ article_ids                 │
        │ ---     ┆ ---                         │
        │ i64     ┆ list[str]                   │
        ╞═════════╪═════════════════════════════╡
        │ 1       ┆ ["News", "Sports"]          │
        │ 2       ┆ ["Sports", "Entertainment"] │
        │ 3       ┆ ["News", null]              │
        │ 4       ┆ [null, null]                │
        │ 5       ┆ [null]                      │
        └─────────┴─────────────────────────────┘
    """
    GROUPBY_ID = generate_unique_name(behaviors.columns, "_groupby_id")
    behaviors: pl.LazyFrame = behaviors.lazy().with_row_count(GROUPBY_ID)
    # =>
    select_column = (
        behaviors.select(pl.col(GROUPBY_ID), pl.col(behaviors_column))
        .explode(behaviors_column)
        .with_columns(pl.col(behaviors_column).replace(mapping, default=None))
    )
    # =>
    if drop_nulls:
        select_column = select_column.drop_nulls()
    elif fill_nulls is not None:
        select_column = select_column.with_columns(
            pl.col(behaviors_column).fill_null(fill_nulls)
        )
    # =>
    select_column = select_column.group_by(GROUPBY_ID).agg(behaviors_column)
    return (
        behaviors.drop(behaviors_column)
        .join(select_column, on=GROUPBY_ID, how="left")
        .drop(GROUPBY_ID)
    )


def sort_and_select(
    df: pl.DataFrame,
    n: int = 5,
    labels_col: str = DEFAULT_LABELS_COL,
    inview_col: str = DEFAULT_INVIEW_ARTICLES_COL,
):
    """Selects the first clicked article and n-1 random articles from the inview articles."""
    # a, b = [], []
    # for i, x in enumerate(df[labels_col]):
    #     idx = np.argsort(x)
    #     idx = np.concatenate((idx[: n - 1], idx[-1:]))
    #     shuffle(idx)
    #     a.append(x[idx])
    #     b.append(df[inview_col][i][idx])

    idx = df[labels_col].list.eval(
        pl.element().arg_sort().take([-1] + list(range(n - 1))).shuffle(), parallel=True
    )
    a = df[labels_col].list.gather(idx)
    b = df[inview_col].list.gather(idx)

    return df.with_columns(
        pl.Series(a).alias(labels_col), pl.Series(b).alias(inview_col)
    )
