# def __len__(self):
    #     return len(self.history)

    # def __getitem__(self, idx):
    #     """NOTE: This uses clever tricks in order to minibatch the data. Taken from the original EBNERD code."""

    #     batch_indices = range(idx * self.batch_size, (idx + 1) * self.batch_size)

    #     article_ids = self.data[idx, DEFAULT_HISTORY_ARTICLE_ID_COL]
    #     print(article_ids)
    #     if isinstance(article_ids, int):
    #         article_ids = [article_ids]

    #     # Select article titles
    #     articles = [
    #         self.articles.filter(pl.col(DEFAULT_ARTICLE_ID_COL).is_in(artids))[
    #             DEFAULT_TITLE_COL
    #         ].to_list()
    #         for artids in article_ids
    #     ]

    #     history = [
    #         article
    #         for article, article_id in zip(articles, article_ids)
    #         if article_id in self.data[idx, 2]
    #     ]

    #     candidate = [
    #         article
    #         for article, article_id in zip(articles, article_ids)
    #         if article_id in self.data[idx, DEFAULT_INVIEW_ARTICLES_COL]
    #     ]

    #     pprint(articles)
    #     encodings = [
    #         self.tokenizer(
    #             article,
    #             padding="max_length",
    #             truncation=True,
    #             max_length=self.max_length,
    #             return_tensors="pt",
    #         )
    #         for article in articles
    #     ]
    #     encoding = {
    #         "input_ids": torch.cat([enc["input_ids"] for enc in encodings]),
    #         "attention_mask": torch.cat([enc["attention_mask"] for enc in encodings]),
    #     }

    #     return encoding
