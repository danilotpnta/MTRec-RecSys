import sys
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
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_IMPRESSION_ID_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_LABELS_COL,
    DEFAULT_SUBTITLE_COL,
    DEFAULT_TITLE_COL,
    DEFAULT_TOPICS_COL,
    DEFAULT_USER_COL,
    DEFAULT_NER_COL,
)

from ebrec.utils._polars import slice_join_dataframes
from ebrec.utils._python import (
    create_lookup_dict,
    create_lookup_dict_,
    print_dict_summary,
    create_lookup_objects,
    create_lookup_objects_,
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
                DEFAULT_ARTICLE_ID_COL,    # article_id
                DEFAULT_TITLE_COL,         # title
                DEFAULT_BODY_COL,          # body
                DEFAULT_SUBTITLE_COL,      # subtitle
                DEFAULT_TOPICS_COL,        # topics
                DEFAULT_CATEGORY_STR_COL,  # category_str
                DEFAULT_NER_COL,           # ner_clusters
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
        }

        np.save(path + "/lookup_matrix.npy", self.lookup_matrix)
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

        dataset.lookup_matrix = np.load(path + "/lookup_matrix.npy")

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
            ).collect()
        )

    def _prepare_training_data(self, embeddings_path=None):
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
             #.pipe(sort_and_select, n=self.max_labels)
            .with_columns(pl.col(DEFAULT_LABELS_COL).list.len().alias(N_SAMPLES_COL))
        )

        assert (
            embeddings_path is not None
        ), "You need to provide a path to the embeddings file."
        embeddings = pl.read_parquet(embeddings_path)

        self.articles = (
            self.articles.lazy()
            .join(embeddings.lazy(), on=DEFAULT_ARTICLE_ID_COL, how="inner")
            .rename({embeddings.columns[-1]: DEFAULT_TOKENS_COL})
            .collect()
        )

        article_dict = create_lookup_dict(
            self.articles.select(DEFAULT_ARTICLE_ID_COL, # KEY
                                 DEFAULT_TOKENS_COL,     # VALUE_1
                                 DEFAULT_NER_COL),       # VALUE_2
            DEFAULT_ARTICLE_ID_COL,
            DEFAULT_TOKENS_COL,
            DEFAULT_NER_COL,
        )
        article_dict = create_lookup_dict_(
            self.articles.select(DEFAULT_ARTICLE_ID_COL, # KEY
                                 DEFAULT_TOKENS_COL,     # VALUE_1
                                 ),       # VALUE_2
            DEFAULT_ARTICLE_ID_COL,
            DEFAULT_TOKENS_COL,
        )
        
        # self.lookup_indexes, self.lookup_matrix, self_ner_matrix = create_lookup_objects(
        #     article_dict, unknown_representation="zeros"
        # )
        self.lookup_indexes, self.lookup_matrix = create_lookup_objects_(
            article_dict, unknown_representation="zeros"
        )

        self.data = self.data.pipe(
            map_list_article_id_to_value,
            behaviors_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
            mapping=self.lookup_indexes,
            fill_nulls=[0],
        ).pipe(
            map_list_article_id_to_value,
            behaviors_column=DEFAULT_INVIEW_ARTICLES_COL,
            mapping=self.lookup_indexes,
            fill_nulls=[0],
        )

        print("self.data" ,self.data)
        '''
        ┌─────────┬─────────────────────┬───────────────┬─────────────┬───────────┬────────────────────────────┬────────────────────────────┐
        │ user_id ┆ article_ids_clicked ┆ impression_id ┆ labels      ┆ n_samples ┆ article_id_fixed           ┆ article_ids_inview         │
        │ ---     ┆ ---                 ┆ ---           ┆ ---         ┆ ---       ┆ ---                        ┆ ---                        │
        │ u32     ┆ list[i32]           ┆ u32           ┆ list[i8]    ┆ u32       ┆ list[list[i64]]            ┆ list[list[i64]]            │
        ╞═════════╪═════════════════════╪═══════════════╪═════════════╪═══════════╪════════════════════════════╪════════════════════════════╡
        │ 22779   ┆ [9759966]           ┆ 48401         ┆ [0, 0, … 1] ┆ 11        ┆ [[9199], [9110], … [9330]] ┆ [[9620], [9719], … [8515]] │
        │ 150224  ┆ [9778661]           ┆ 152513        ┆ [0, 0, … 0] ┆ 17        ┆ [[9101], [9135], … [6139]] ┆ [[3193], [5501], … [2799]] │
        │ 160892  ┆ [9777856]           ┆ 155390        ┆ [0, 0, … 0] ┆ 11        ┆ [[8463], [9008], … [9301]] ┆ [[9988], [9884], … [9996]] │
        │ 1001055 ┆ [9776566]           ┆ 214679        ┆ [0, 0, … 1] ┆ 9         ┆ [[9082], [9069], … [9285]] ┆ [[9862], [9807], … [9853]] │
        │ 1001055 ┆ [9776553]           ┆ 214681        ┆ [0, 0, … 0] ┆ 18        ┆ [[9082], [9069], … [9285]] ┆ [[9707], [9807], … [9843]] │
        │ …       ┆ …                   ┆ …             ┆ …           ┆ …         ┆ …                          ┆ …                          │
        │ 2053999 ┆ [9775562]           ┆ 579983230     ┆ [0, 0, … 0] ┆ 19        ┆ [[8815], [8788], … [9303]] ┆ [[9664], [9107], … [2457]] │
        │ 2053999 ┆ [9775361]           ┆ 579983231     ┆ [1, 0, … 0] ┆ 37        ┆ [[8815], [8788], … [9303]] ┆ [[9718], [9738], … [9723]] │
        │ 2060487 ┆ [9775699]           ┆ 579984721     ┆ [0, 0, … 0] ┆ 5         ┆ [[9247], [9247], … [9261]] ┆ [[9747], [9758], … [9714]] │
        │ 2060487 ┆ [9758424]           ┆ 579984723     ┆ [0, 0, … 0] ┆ 20        ┆ [[9247], [9247], … [9261]] ┆ [[9694], [9746], … [9727]] │
        │ 2096611 ┆ [9770369]           ┆ 580097289     ┆ [0, 1, … 0] ┆ 6         ┆ [[788], [8699], … [9131]]  ┆ [[4614], [9313], … [7179]] │
        └─────────┴─────────────────────┴───────────────┴─────────────┴───────────┴────────────────────────────┴────────────────────────────┘
        '''
        print(self.data.explode('article_id_fixed')[0])
        self.data = PolarsDataFrameWrapper(self.data)
       
        sys.exit()

    def create_category_labels(self):       

        unique_categories = self.df_articles.select("category_str").unique().to_series().to_list()
        len_categories = len(unique_categories)

        # Do one-hot encoding dictionary for the categories
        cat_names_dic = {cat_name: [0]*len_categories for cat_name in unique_categories}
        for i, cat_name in enumerate(unique_categories):
            cat_names_dic[cat_name][i] = 1

        def map_category_to_vector(category_str):
            return cat_names_dic[category_str]

        self.df_articles = self.df_articles.with_columns(
                pl.col('category_str').apply(map_category_to_vector).alias('category_vector')
        )

        article_id_to_vector = {row[0]: row[1] for row in zip(self.df_articles['article_id'], self.df_articles['category_vector'])}

        def generate_labels(self, column_name):
            labels = []
            for list_articles_ids in self.df_behaviors[column_name]:
                vectors = []
                for id in list_articles_ids:
                    vectors.append(article_id_to_vector.get(id, [0] * len_categories))
                labels.append(vectors)
            return np.array(labels)

        self.cat_labels_bh_inview = generate_labels('article_ids_inview')
        self.cat_labels_bh_fixed = generate_labels('article_id_fixed')


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
        history_input = self.lookup_matrix[
            batch[DEFAULT_HISTORY_ARTICLE_ID_COL].to_list()
        ]
        candidate_input = self.lookup_matrix[
            batch[DEFAULT_INVIEW_ARTICLES_COL].to_list()
        ]

        # =>
        # ner_input = candidate_input = self.ner_matrix[
        #     batch[DEFAULT_NER_COL].to_list()
        # ]
        
        # print(ner_input)
        # sys.exit()
        # =>
        labels_item = np.array(batch[DEFAULT_LABELS_COL][0])
        idx = np.argsort(labels_item)
        pos_idx_start = list(labels_item[idx]).index(1)
        pos_idxs = np.random.choice(idx[pos_idx_start:], size=(1,), replace=False)
        population = idx[:pos_idx_start]
        population_size = len(population)
        sample_size = self.max_labels - 1

        if population_size < sample_size:
            neg_idxs = np.random.choice(population, size=sample_size, replace=True)
        else:
            neg_idxs = np.random.choice(population, size=sample_size, replace=False)
    
        idx = np.concatenate((neg_idxs, pos_idxs))
        #shuffle(idx)
        history_input = torch.tensor(history_input).squeeze().bfloat16()
        candidate_input = torch.tensor(candidate_input[0][idx]).squeeze().bfloat16()
        y = torch.tensor(labels_item[idx], dtype=torch.bfloat16).squeeze()
        # ========================
        return history_input, candidate_input, y


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
        self.data_path = data_path
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
        savefolder = os.path.join(self.data_path, self.dataset)
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
        embeddings_folder = os.path.join(
            self.data_path.rpartition("/")[0], self.embeddings
        )
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

    def setup(self, stage: str):
        match stage:
            case "fit" | "validation" | None:
                # Load the training data
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
                raise NotImplementedError("Test data not implemented yet.")

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
    behaviors: pl.DataFrame,
    behaviors_column: str,
    mapping: "dict[int, pl.Series]",
    drop_nulls: bool = False,
    fill_nulls: any = None,
) -> pl.DataFrame:
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
    behaviors = behaviors.lazy().with_row_count(GROUPBY_ID)
    # =>
    select_column = (
        behaviors.select(pl.col(GROUPBY_ID), pl.col(behaviors_column))
        .explode(behaviors_column)
        .with_columns(pl.col(behaviors_column).replace(mapping, default=None))
        .collect()
    )
    # =>
    if drop_nulls:
        select_column = select_column.drop_nulls()
    elif fill_nulls is not None:
        select_column = select_column.with_columns(
            pl.col(behaviors_column).fill_null(fill_nulls)
        )
    # =>
    select_column = (
        select_column.lazy().group_by(GROUPBY_ID).agg(behaviors_column).collect()
    )
    return (
        behaviors.drop(behaviors_column)
        .collect()
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
    a, b = [], []
    for i, x in enumerate(df[labels_col]):
        idx = np.argsort(x)
        idx = np.concatenate((idx[: n - 1], idx[-1:]))
        shuffle(idx)
        a.append(x[idx])
        b.append(df[inview_col][i][idx])

    return df.with_columns(
        pl.Series(a).alias(labels_col), pl.Series(b).alias(inview_col)
    )
