import logging as log
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from typing import Optional
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import torch


def collate_fn(batch):

    batch = batch
    pass


class TransformersTextClassDatamodule(pl.LightningDataModule):
    def __init__(self, args,
                docs,
                labels,
                word2idx,
                train_mask,
                val_mask,
                test_mask,
                 **kwargs):
        super().__init__()

        self.model_name = args.transformer_model_name # used to initialize the right tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.docs = docs
        self.train_mask = train_mask[len(word2idx):]
        self.val_mask = val_mask[len(word2idx):]
        self.test_mask = test_mask[len(word2idx):]
        self.labels = labels
        self.batch_size = args.batch_size
        self.word2idx = word2idx # used to filter masks

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None


    def _docs_labels_to_df(self, docs, labels):

        labels = torch.LongTensor(labels)
        for i in range(len(docs)):
            docs[i]['labels'] = labels[i]

        return docs
        df = pd.DataFrame({'x': docs, 'y':labels})
        return df

    def setup(self, stage: Optional[str] = None):

        # filter the docs per set
        train_docs = [' '.join(doc) for doc, mask in zip(self.docs, self.train_mask) if mask]
        train_docs = [self.tokenizer(doc, truncation=True, padding=True, ) for doc in train_docs]
        train_labels = [label for label, mask in zip(self.labels, self.train_mask) if mask]

        val_docs = [' '.join(doc) for doc, mask in zip(self.docs, self.val_mask) if mask]
        val_docs = [self.tokenizer(doc, truncation=True) for doc in val_docs]
        val_labels = [label for label, mask in zip(self.labels, self.val_mask) if mask]

        test_docs = [' '.join(doc) for doc, mask in zip(self.docs, self.test_mask) if mask]
        test_docs = [self.tokenizer(doc, truncation=True) for doc in test_docs]
        test_labels = [label for label, mask in zip(self.labels, self.test_mask) if mask]

        # preprocess the samples
        self._train_dataset = self._docs_labels_to_df(train_docs, train_labels)
        self._val_dataset = self._docs_labels_to_df(val_docs, val_labels)
        self._test_dataset = self._docs_labels_to_df(test_docs, test_labels)


    def train_dataloader(self) -> DataLoader:
        """Function that loads the train set."""
        return DataLoader(
            dataset=self._train_dataset,
            # sampler=RandomSampler(self._train_dataset),
            batch_size=self.batch_size,
            # collate_fn=collate_fn,
            collate_fn=DataCollatorWithPadding(tokenizer=self.tokenizer),
            num_workers=0,
        )

    def val_dataloader(self) -> DataLoader:
        """Function that loads the validation set."""
        return DataLoader(
            dataset=self._val_dataset,
            batch_size=self.batch_size,
            # collate_fn=collate_fn,
            collate_fn=DataCollatorWithPadding(tokenizer=self.tokenizer),
            num_workers=0,
        )

    def test_dataloader(self) -> DataLoader:
        """Function that loads the test set."""
        return DataLoader(
            dataset=self._test_dataset,
            batch_size=self.batch_size,
            # collate_fn=collate_fn,
            collate_fn=DataCollatorWithPadding(tokenizer=self.tokenizer),
            num_workers=0,
        )

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("TransformersTextClassDataset")

        parser.add_argument(
            "--batch_size",
            type=int,
            default=16,
            help="Batch size.",
        )

        return parent_parser