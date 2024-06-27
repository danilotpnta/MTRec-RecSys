import polars as pl
from mind.dataset import (
    MINDDataset,
    COL_IMPRESSION_ID,
    COL_USER_ID,
    COL_TIME,
    COL_HISTORY,
    COL_IMPRESSIONS,
    COL_NEWS_ID,
    COL_CATEGORY,
    COL_SUBCATEGORY,
    COL_TITLE,
    COL_ABSTRACT,
    COL_URL,
    COL_TITLE_ENTITIES,
    COL_ABSTRACT_ENTITIES,
)
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from recsys.model import BERTMultitaskRecommender
from pytorch_lightning import Trainer

BATCH_SIZE = 16

behaviors = pl.read_csv('data/MINDsmall_train/behaviors.tsv', separator='\t', new_columns=[COL_IMPRESSION_ID, COL_USER_ID, COL_TIME, COL_HISTORY, COL_IMPRESSIONS], has_header=False)

articles = pl.read_csv('data/MINDsmall_train/news.tsv', separator='\t', has_header=False, new_columns=[COL_NEWS_ID, COL_CATEGORY, COL_SUBCATEGORY, COL_TITLE, COL_ABSTRACT, COL_URL, COL_TITLE_ENTITIES, COL_ABSTRACT_ENTITIES])

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

dataset = MINDDataset(behaviors, articles, tokenizer)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = BERTMultitaskRecommender(
    epochs=1,
    lr=2e-5,
    wd=1e-4,
    batch_size=BATCH_SIZE,
    steps_per_epoch=len(dataset)//BATCH_SIZE,
    use_gradient_surgery=True,
    n_categories=dataset.max_categories,
    sentiment_labels=dataset.max_sentiment_labels,
    use_lora=True,
    disable_category=False,
    disable_sentiment=True,
)

trainer = Trainer(max_epochs=10, num_sanity_val_steps=1, log_every_n_steps=1)

trainer.fit(model, dataloader, dataloader)