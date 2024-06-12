import os.path
from math import ceil
from typing import Any

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
)
from ebrec.utils._polars import slice_join_dataframes
from ebrec.utils._python import (
    create_lookup_dict,
    create_lookup_objects,
    generate_unique_name,
    repeat_by_list_values_from_matrix,
)
from torch.utils.data import Dataset

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
        padding_value: int = 0,
        max_length=128,
        batch_size=32,
        embeddings_path=None,
        neg_count=5,
    ):
        self.behaviors = behaviors
        self.history = history
        self.articles = articles

        self.history_size = history_size
        self.padding_value = padding_value
        self.batch_size = batch_size
        self.neg_count = neg_count

        # self.tokenizer = tokenizer
        # self.max_length = max_length

        # TODO: (Matey:) Decided to instead only use pre-computed embeddings for now. You might want to look into this later down the line and implement custom embeddings (and e.g. train BERT as well).
        self.embeddings_path = embeddings_path

        # NOTE: Keep an eye on this if memory issues arise, e.g. if you need lazy loading or something similar.
        self.articles = self.articles.select(
            [
                DEFAULT_ARTICLE_ID_COL,
                DEFAULT_TITLE_COL,
                DEFAULT_BODY_COL,
                DEFAULT_SUBTITLE_COL,
                DEFAULT_TOPICS_COL,
                DEFAULT_CATEGORY_STR_COL,
            ]
        ).collect()

        self._process_history()
        self._set_data()

    def _process_history(self):
        self.history = (
            self.history.select(DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL)
            .pipe(
                truncate_history,
                column=DEFAULT_HISTORY_ARTICLE_ID_COL,
                history_size=self.history_size,
                padding_value=self.padding_value,
                enable_warning=False,
            )
            .collect()
        )

    def _set_data(self):
        self.behaviors = self.behaviors.collect()
        self.data: pl.DataFrame = (
            slice_join_dataframes(
                self.behaviors,
                self.history,
                on=DEFAULT_USER_COL,
                how="left",
            )
            .select(COLUMNS)
            .pipe(
                create_binary_labels_column,
                seed=42,
                clicked_col=DEFAULT_CLICKED_ARTICLES_COL,
                inview_col=DEFAULT_INVIEW_ARTICLES_COL,
                label_col=DEFAULT_LABELS_COL,
            )
        ).with_columns(pl.col(DEFAULT_LABELS_COL).list.len().alias(N_SAMPLES_COL))

        assert (
            self.embeddings_path is not None
        ), "You need to provide a path to the embeddings file."

        embeddings = pl.scan_parquet(self.embeddings_path)

        self.articles = (
            self.articles.lazy()
            .join(embeddings, on=DEFAULT_ARTICLE_ID_COL, how="inner")
            .rename({embeddings.columns[-1]: DEFAULT_TOKENS_COL})
            .collect()
        )

        article_dict = create_lookup_dict(
            self.articles.select(DEFAULT_ARTICLE_ID_COL, DEFAULT_TOKENS_COL),
            key=DEFAULT_ARTICLE_ID_COL,
            value=DEFAULT_TOKENS_COL,
        )

        self.lookup_indexes, self.lookup_matrix = create_lookup_objects(
            article_dict, unknown_representation="zeros"
        )

    def __len__(self):
        """
        Number of batch steps in the data
        """
        return ceil(self.behaviors.shape[0] / self.batch_size)

    def __getitem__(self, index: int):
        """
        Get the batch of samples for the given index.

        Note: The dataset class provides a single index for each iteration. The batching is done internally in this method
        to utilize and optimize for speed. This can be seen as a mini-batching approach.

        Args:
            index (int): An integer index.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the input features and labels as torch Tensors.
                Note, the output of the PyTorch DataLoader is (1, *shape), where 1 is the DataLoader's batch_size.
        """
        # Clever way to batch the data:
        batch_indices = range(index * self.batch_size, (index + 1) * self.batch_size)
        batch = self.data[batch_indices]

        x = (
            batch.drop(DEFAULT_LABELS_COL)
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
        )
        # =>
        repeats = np.array(batch[N_SAMPLES_COL])
        # =>
        history_input = repeat_by_list_values_from_matrix(
            input_array=x[DEFAULT_HISTORY_ARTICLE_ID_COL].to_list(),
            matrix=self.lookup_matrix,
            repeats=repeats,
        ).squeeze(2)
        # =>
        candidate_input = self.lookup_matrix[
            x[DEFAULT_INVIEW_ARTICLES_COL].explode().to_list()
        ]
        # =>
        history_input = torch.tensor(history_input)
        candidate_input = torch.tensor(candidate_input)
        y = batch[DEFAULT_LABELS_COL].explode()
        filter = np.array([np.concatenate((
            np.random.choice(np.nonzero(y[end_idx-repeats[i]:end_idx] == 1)[0], size=(1,)), np.random.choice(np.nonzero(y[end_idx-repeats[i]:end_idx] == 0)[0], size=self.neg_count)
        )) for i, end_idx in enumerate(np.cumsum(repeats))]).flatten()
        y = torch.tensor(y[filter]).float()
        candidate_input = candidate_input[filter]
        # ========================
        return history_input, candidate_input, y, repeats

class NewsDatasetV2(NewsDataset):
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, index: int):
        sample = self.data[index]
        sample = sample.pipe(
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

        _history = sample[DEFAULT_HISTORY_ARTICLE_ID_COL].explode().explode().to_list()
        history = torch.from_numpy(self.lookup_matrix[_history])
        _candidates = sample[DEFAULT_INVIEW_ARTICLES_COL].explode().explode().to_list()
        candidates = torch.from_numpy(self.lookup_matrix[_candidates])
        # dataset.lookup_indexes
        labels = torch.tensor(sample[DEFAULT_LABELS_COL].to_list()[0])
        return history, candidates, labels

def load_data(
    tokenizer: Any, data_path: str, split="train", embeddings_path: str = None, batch_size=32
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
    embeddings_path : str, optional
        Path to the embeddings file, by default None

    Returns
    -------
    NewsDataset
        The dataset containing the data.
    """
    _data_path = os.path.join(data_path, split)

    df_behaviors = pl.scan_parquet(_data_path + "/behaviors.parquet")
    df_history = pl.scan_parquet(_data_path + "/history.parquet")
    df_articles = pl.scan_parquet(data_path + "/articles.parquet")

    return NewsDataset(
        tokenizer,
        df_behaviors,
        df_history,
        df_articles,
        embeddings_path=embeddings_path,
        batch_size=batch_size,
    )


def map_list_article_id_to_value(
    behaviors: pl.DataFrame,
    behaviors_column: str,
    mapping: dict[int, pl.Series],
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
    behaviors = behaviors.lazy().with_row_index(GROUPBY_ID)
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
