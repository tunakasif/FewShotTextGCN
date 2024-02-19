import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification

from optimizers.ranger21 import Ranger21


class LitTransformer(pl.LightningModule):
    def __init__(self, args, num_labels, **kwargs):
        super().__init__()
        self.save_hyperparameters(args)

        self.max_epochs = args.max_epochs
        self.num_labels = num_labels
        self.model_name = args.transformer_model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=num_labels)
        self.optimizer_type = args.optimizer_type

    @staticmethod
    def _get_accuracy(logits, y_true):

        batch_size = logits.size(0)
        y_pred = torch.argmax(logits, dim=-1)

        num_correct = (y_pred == y_true).sum().float()

        return num_correct / batch_size


    def forward(self, x):

        return self.model(**x)

    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = out.loss
        accuracy = self._get_accuracy(logits=out.logits, y_true=batch.labels)
        self.log('train_loss', loss)
        self.log('train_acc', accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        loss = out.loss
        accuracy = self._get_accuracy(logits=out.logits, y_true=batch.labels)
        self.log('val_loss', loss)
        self.log('val_acc', accuracy)
        return loss

    def test_step(self, batch, batch_idx):
        out = self(batch)
        loss = out.loss
        accuracy = self._get_accuracy(logits=out.logits, y_true=batch.labels)
        self.log('test_loss', loss)
        self.log('test_acc', accuracy)
        return loss

    def configure_optimizers(self):
        lr = 0.001 if self.hparams.lr == "auto" else float(self.hparams.lr)
        if self.optimizer_type.lower() == "ranger":
            optimizer = Ranger21(
                self.parameters(),
                lr=lr,
                num_epochs=self.max_epochs,
                num_batches_per_epoch=1,  # TODO: adapt
                warmdown_min_lr=1e-6,
                weight_decay=1e-4,
            )
        elif self.optimizer_type == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        else:
            raise NotImplementedError()

        return optimizer

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("TransformersTextClassDataset")

        parser.add_argument(
            "--transformer_model_name",
            type=str,
            default='bert-base-multilingual-cased',
            help="Model name (to download from HF) or path to load locally",
        )
        parser.add_argument(
            "--lr",
            type=str,
            default='2e-5',
            help="Model name (to download from HF) or path to load locally",
        )
        parser.add_argument(
            "--optimizer_type",
            default="adam",
            help="Optimizer to use",
        )


        return parent_parser