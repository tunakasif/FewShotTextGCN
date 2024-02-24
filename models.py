import abc
import os
import logging
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_cluster import random_walk
from torch_geometric.nn import (
    MessagePassing,
    GCNConv,
    GATConv,
    Sequential,
    SAGEConv,
    SGConv,
)
from torch_geometric.utils import (
    remove_self_loops,
    add_self_loops,
    softmax,
    k_hop_subgraph,
    dropout_adj,
)
from torch_geometric.nn.inits import glorot, zeros
import pytorch_lightning as pl
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.distances import CosineSimilarity

from optimizers.ranger21 import Ranger21
from sklearn.metrics import matthews_corrcoef

# From: https://raw.githubusercontent.com/codeKgu/Text-GCN/master/model_text_gnn.py
EPS = 1e-15

logger = logging.getLogger(__name__)


class UnsupGNN(pl.LightningModule, abc.ABC):
    # Node2vec explained: https://towardsdatascience.com/node2vec-explained-graphically-749e49b7eb6b
    # Unsupervised learning: https://github.com/pyg-team/pytorch_geometric/issues/64#issuecomment-505734268
    # Possible complete implementation: https://towardsdatascience.com/pytorch-geometric-graph-embedding-da71d614c3a
    def __init__(
        self,
        args,
        num_labels,
        num_features,
        class_weights=None,
        num_subwords=None,
        num_doc_nodes=None,
    ):
        super().__init__()
        self.save_hyperparameters(args)
        self.calc_mcc = args.calc_mcc
        if self.calc_mcc:
            print("Using MCC")
        self.args = args
        self.num_labels = num_labels
        self.num_features = num_features
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.lr = args.lr
        self.bn = args.bn
        self.optimizer_type = args.optimizer_type
        self.aggregate_subwords = (
            args.use_bbpe
        )  # Add a module to the model that takes in subwords and aggregates to word
        self.subword_aggregator_type = args.subword_aggregator_type
        self.class_weights = class_weights
        self.num_subwords = num_subwords
        self.num_doc_nodes = num_doc_nodes

        # Self training
        self.self_training = args.use_self_training
        self.self_training_conf_thresh = args.self_training_conf_thresh

        # Unsup training args
        self.use_unsup_loss = args.use_unsup_loss
        self.walks_per_node = args.walks_per_node
        self.walk_length = args.walk_length
        self.num_negative_samples = args.num_negative_samples
        # Random walk parameters
        # TODO:
        self.p = args.rw_inverse_return_likelihood  # inverse return likelihood
        self.q = (
            args.rw_inverse_bfs_likelihood
        )  # 0.001 # inverse depth-first likelihood

        if self.aggregate_subwords:
            self.subword_aggregator = self.init_subword_aggregator()

        self.convs, self.class_head = self.init_model(
            num_features=num_features,
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            num_labels=num_labels,
        )
        if self.bn:
            self.batch_norms = nn.ModuleList(
                [nn.BatchNorm1d(self.hidden_dim) for i in range(self.num_layers)]
            )

        self.loss = nn.CrossEntropyLoss(weight=class_weights)

    def init_subword_aggregator(self):

        subword_emb = nn.Embedding(
            self.num_subwords + self.num_doc_nodes + 1, self.hidden_dim
        )

        modules = [subword_emb]

        if self.subword_aggregator_type == "transformer":
            contextualizer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=4,
                dim_feedforward=2 * self.hidden_dim,
                batch_first=True,
            )
            modules.append(contextualizer)
        if self.subword_aggregator_type == "gru":
            contextualizer = nn.GRU(
                input_size=self.hidden_dim,
                hidden_size=self.hidden_dim,
                num_layers=1,
                batch_first=True,
            )
            modules.append(contextualizer)

        subword_aggregator = nn.Sequential(*modules)

        return subword_aggregator

    @abc.abstractmethod
    def init_model(self, num_features, num_layers, hidden_dim, num_labels):
        pass

    def configure_optimizers(self):
        # Start with simple Adam optimizer

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

    def get_pseudo_label_loss(self, mask, y_pred):


        mask_adapted = mask.detach().clone()
        self_train_conf_thresh = self.self_training_conf_thresh
        # num samples
        num_unmask = mask.sum()
        num_classes = y_pred.size(-1)
        num_per_class = num_unmask / num_classes

        num_pred_per_class = torch.zeros(num_classes, device=mask.device)
        probs = torch.softmax(y_pred, dim=-1)
        _, y_true_pseudo = torch.max(y_pred, dim=1)
        for i in range(num_classes):
            class_probs = probs[:, i]
            # Filter out low confidence
            satisfies_mask = class_probs > self_train_conf_thresh
            # Filter out nodes in train and test set
            satisfies_mask *= ~mask
            satisfies_nodes = satisfies_mask.nonzero().squeeze()
            satisfies_doc_nodes = satisfies_nodes[
                satisfies_nodes > self.doc_node_start_index
            ]
            num_pred_per_class[i] = satisfies_doc_nodes.size(0)

            mask_adapted[satisfies_doc_nodes] = torch.ones_like(satisfies_doc_nodes) > 0

        if num_pred_per_class.sum() > 0:


            class_weights = torch.zeros_like(num_pred_per_class)
            class_weights[num_pred_per_class > 0] = (
                num_pred_per_class[num_pred_per_class > 0] ** -1
            )
            pseudo_label_loss = F.cross_entropy(
                y_pred[mask_adapted], y_true_pseudo[mask_adapted], weight=class_weights
            )

        else:
            pseudo_label_loss = 0

        self.log("pseudo_label_loss", pseudo_label_loss)

        return pseudo_label_loss

        # return y_true_adapted, mask_adapted

    def forward(self, batch, mode="train"):
        emb = batch.x
        # Dropout in node features
        emb = F.dropout(emb, p=self.dropout, training=self.training)
        # Dropedge for dropping edges within graph
        edge_index, edge_attr = dropout_adj(
            batch.edge_index, batch.edge_attr, p=self.dropout, training=self.training
        )
        for idx, conv in enumerate(self.convs):

            emb = F.gelu(conv(emb, edge_index, edge_weight=edge_attr))  # [idx]))
            if self.bn:
                emb = self.batch_norms[idx](emb)

            # Only calculate the loss on the nodes corresponding to the mask
            if mode == "train":
                mask = batch.train_mask
            elif mode == "val":
                mask = batch.val_mask
            elif mode == "test":
                mask = batch.test_mask
            else:
                assert False, f"Unknown forward mode: {mode}"

        y_pred = self.class_head(
            F.dropout(
                emb, p=self.dropout, training=self.training
            )  # Only apply dropout for classification, not on the embeddings themselves for better unsup learning
        )

        loss = self.loss(y_pred[mask], batch.y[mask])
        if self.calc_mcc:
            mcc = matthews_corrcoef(y_pred=y_pred[mask].argmax(dim=-1), y_true=batch.y[mask])
            return loss, mcc, emb
        acc = (y_pred[mask].argmax(dim=-1) == batch.y[mask]).float().mean()
        return loss, acc, emb

    @abc.abstractmethod
    def pos_sample(self, batch, start_node):
        """Sample positive nodes for a given start node for unsupervised learning"""
        pass

    @abc.abstractmethod
    def neg_sample(self, batch, start_node):
        """Sample negative nodes for a given start node for unsupervised learning"""
        pass

    def unsup_loss(self, pos_rw, neg_rw):
        # from: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/node2vec.html
        # Positive loss.
        h_start, h_rest = pos_rw[:, 0, :].contiguous(), pos_rw[:, 1, :].contiguous()

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + EPS).mean()

        # Negative loss.
        h_start, h_rest = neg_rw[:, 0, :].contiguous(), neg_rw[:, 1, :].contiguous()
        out = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + EPS).mean()

        return pos_loss + neg_loss

    def unsup_step(self, batch, emb):
        # batch is complete sampled graph

        start_node = self.get_start_node_rw(batch)  # row[0]
        pos_rw = self.pos_sample(batch, start_node)
        neg_rw = self.neg_sample(batch, start_node)

        # Replace node indices by their respective embeddings
        pos_rw = F.embedding(pos_rw, emb)
        neg_rw = F.embedding(neg_rw, emb)

        return self.unsup_loss(pos_rw, neg_rw)

    @abc.abstractmethod
    def get_start_node_rw(self, batch):
        pass

    def training_step(self, batch, batch_idx):
        loss, acc, emb = self.forward(batch, mode="train")

        unsup_loss = 0
        if self.use_unsup_loss:
            unsup_loss = self.unsup_step(batch, emb)
            self.log("unsup_loss", unsup_loss, on_epoch=True)

        self.log("train_sup_loss", loss, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True)

        return loss + unsup_loss

    def validation_step(self, batch, batch_idx):
        loss, acc, emb = self.forward(batch, mode="val")
        unsup_loss = 0
        if self.use_unsup_loss:
            unsup_loss = self.unsup_step(batch, emb)
            self.log("val_unsup_loss", unsup_loss)

        self.log("val_acc", acc)
        self.log("val_loss", loss + unsup_loss)  # TODO: change

    def test_step(self, batch, batch_idx):
        predictions_correct_or_not, acc, _ = self.forward(batch, mode="test")
        self.log("test_acc", acc)

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("LitSage")

        parser.add_argument(
            "--lr",
            type=str,
            default="auto",
            help="Initial learning rate. Set to 'auto' to use the learning rate finder.",
        )
        parser.add_argument(
            "--optimizer_type",
            default="adam",
            help="Optimizer to use",
        )

        parser.add_argument(
            "--node_embedding_type",
            default="gcn",
            help="Optimizer to use",
        )
        parser.add_argument(
            "--act_fn",
            default="prelu",
            help="Optimizer to use",
        )
        parser.add_argument(
            "--bn",
            action="store_false",
            help="Use batchnorm",
        )
        parser.add_argument(
            "--hidden_dim",
            default=128,
            type=int,
            help="Hidden dimension",
        )
        parser.add_argument(
            "--dropout",
            default=0.2,
            type=float,
            help="Dropout",
        )
        parser.add_argument(
            "--num_layers",
            default=2,
            type=int,
            help="Number of node embedding layers",
        )
        # Unsup training args
        parser.add_argument(
            "--use_unsup_loss",
            action="store_true",
            help="Use unsupervised loss",
        )
        parser.add_argument(
            "--walks_per_node",
            default=200,
            type=int,
            help="Hidden dimension",
        )
        parser.add_argument(
            "--walk_length",
            default=2,
            type=int,
            help="Hidden dimension",
        )
        parser.add_argument(
            "--num_negative_samples",
            default=50,
            type=int,
            help="Hidden dimension",
        )
        parser.add_argument(
            "--rw_inverse_return_likelihood",
            default=100000,
            type=float,
            help="p value from Node2Vec paper",
        )
        parser.add_argument(
            "--rw_inverse_bfs_likelihood",
            default=0.00001,
            type=float,
            help="q value from Node2Vec paper",
        )
        # Subwords
        parser.add_argument(
            "--subword_aggregator_type",
            type=str,
            help="Kind of subword aggregator to use. Currently supported: ['transformer', 'gru', 'none']",
        )
        # Self-training args
        parser.add_argument(
            "--use_self_training",
            action="store_true",
            help="Use unsupervised loss",
        )
        parser.add_argument(
            "--self_training_conf_thresh",
            default=0.5,
            type=float,
            help="q value from Node2Vec paper",
        )
        parser.add_argument(
            "--calc_mcc",
            default=False,
            type=bool,
            help="Calculate Matthews correlation instead of accuracy",
        )

        return parent_parser


class CustomDocGraphGNN(UnsupGNN):
    # Node2vec explained: https://towardsdatascience.com/node2vec-explained-graphically-749e49b7eb6b
    # Unsupervised learning: https://github.com/pyg-team/pytorch_geometric/issues/64#issuecomment-505734268
    # Possible complete implementation: https://towardsdatascience.com/pytorch-geometric-graph-embedding-da71d614c3a
    def __init__(
        self,
        args,
        num_labels,
        num_features,
        doc_node_start_index,
        class_weights=None,
        num_subwords=None,
        num_doc_nodes=None,
    ):
        super().__init__(
            args,
            num_labels,
            num_features,
            class_weights=class_weights,
            # doc_node_start_index=doc_node_start_index,
            num_subwords=num_subwords,
            num_doc_nodes=num_doc_nodes,
        )

        self.tsa_schedule = args.tsa_schedule
        self.max_epochs = args.max_epochs
        self.doc_node_start_index = doc_node_start_index
        # TODO: play around seems to result in deterministic behavior
        self.p = 1  # 100000
        self.q = 1  # 0.00001

        self.neg_samples_per_word = None

        self.use_most_similar_docs = args.use_most_similar_docs

        self.projector_labels = []
        self.log_emb = False
        self.use_triplet_loss = args.use_triplet_loss
        self.triplet_loss_margin = args.triplet_margin

        self.triplet_distance = (
            CosineSimilarity() if args.triplet_distance == "cosine" else None
        )

        self.triplet_loss = TripletMarginLoss(
            margin=self.triplet_loss_margin, distance=self.triplet_distance
        )
        self.triplet_dummy_labels = torch.arange(num_features)

    def init_model(self, num_features, num_layers, hidden_dim, num_labels):
        convs = nn.ModuleList()
        first_hidden_dim = hidden_dim if self.aggregate_subwords else num_features
        # GCN conv is already initialized using xavier/glorot
        convs.append(GCNConv(first_hidden_dim, hidden_dim, cached=True))

        for _ in range(num_layers - 1):
            convs.append(GCNConv(hidden_dim, hidden_dim, cached=True))

        class_head = nn.Linear(hidden_dim, num_labels)
        nn.init.xavier_uniform_(class_head.weight)

        return convs, class_head

    def forward(self, batch, mode="train"):

        # Input features of graph
        if self.aggregate_subwords:
            emb = self.subword_aggregator(batch.x)
            if self.subword_aggregator_type == "gru":
                emb = emb[1]
            else:
                emb, _ = torch.max(emb, dim=1)
            # Dropout in node features
            # emb = F.dropout(emb, p=self.dropout, training=self.training)
        else:
            emb = batch.x

        # Dropedge for dropping edges within graph
        edge_index, edge_attr = dropout_adj(
            batch.edge_index, batch.edge_attr, p=self.dropout, training=self.training
        )

        # Graph feature learning
        for idx, conv in enumerate(self.convs):
            emb = conv(emb, edge_index, edge_weight=edge_attr)  # [idx]))
            if idx < len(self.convs) - 1:
                emb = F.gelu(emb)
            else:
                emb = F.tanh(emb)


        y_true = batch.y.clone()
        # Only calculate the loss on the nodes corresponding to the mask
        if mode == "train":
            mask = batch.train_mask

        elif mode == "val":
            mask = batch.val_mask
        elif mode == "test":
            mask = batch.test_mask
        else:
            assert False, f"Unknown forward mode: {mode}"

        # Node classification
        y_pred = self.class_head(F.dropout(emb, p=self.dropout, training=self.training))
        # Self training
        pseudo_label_loss = 0
        if self.self_training and mode == "train":
            pseudo_label_loss = self.get_pseudo_label_loss(batch.test_mask+batch.train_mask, y_pred)

        y_pred, y_true = y_pred[mask], y_true[mask]
        # Compute accuracy on all nodes
        acc = (y_pred.argmax(dim=-1) == y_true).float().mean()
        if self.calc_mcc:
            mcc = matthews_corrcoef(y_pred=y_pred.argmax(dim=-1), y_true=y_true)

        # Filter out very certain preds using TSA and compute loss on remaining nodes
        if self.use_unsup_loss and mode == "train":
            y_pred, y_true = self.mask_predictions_for_tsa(y_pred, y_true)

        if y_pred.nelement() == 0:  # All predictions filtered out by TSA
            loss = y_pred.new_zeros(1)
        else:
            loss = self.loss(y_pred, y_true)

        # Add self-training loss
        loss += pseudo_label_loss

        if self.calc_mcc:
            acc = mcc

        if mode == "test":
            return (y_pred.argmax(dim=-1) == y_true), acc, emb
        else:
            return loss, acc, emb

    def training_step(self, batch, batch_idx):
        loss, acc, emb = self.forward(batch, mode="train")

        if self.log_emb:
            self.logger.experiment[0].add_embedding(
                emb, batch.y, global_step=self.current_epoch, tag="node_emb_final_layer"
            )

        unsup_loss = 0
        if self.use_unsup_loss:
            unsup_loss = self.unsup_step(batch, emb)
            self.log("unsup_loss", unsup_loss, on_epoch=True)

        self.log("train_sup_loss", loss, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True)

        # return unsup_loss
        return loss + unsup_loss

    def validation_step(self, batch, batch_idx):
        loss, acc, emb = self.forward(batch, mode="val")

        self.log("val_acc", acc)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        predictions_correct_or_not, acc, _ = self.forward(batch, mode="test")

        binary_indicator = predictions_correct_or_not.detach().cpu().numpy().astype(int)
        with open(
            os.path.join(self.hparams.experiment_name, "test_binary_indicator.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            for b in binary_indicator:
                f.write(str(b) + "\n")
        self.log("test_acc", acc)

    def _get_tsa_alpha(self):

        total_its = self.max_epochs
        if self.tsa_schedule == "exp":
            alpha = np.exp((self.global_step / total_its - 1) * 5)
        elif self.tsa_schedule == "linear":
            alpha = self.global_step / total_its
        elif self.tsa_schedule == "log":  # log schedule
            alpha = 1 - np.exp(-(self.global_step / total_its) * 5)
        else:
            alpha = None

        return alpha

    def mask_predictions_for_tsa(self, preds, y_true, mask=None):
        """
        Training Signal Annealing
        Masks the predictions and ground truth labels using an exponential schedule.
        alpha = exp((t/T-1)*5), with T the total time steps and t the current step.
        eta = alpha * (1 - (1/K)) + (1/K)
        Only samples with certainty < max_certainty are used for training.
        """
        total_its = self.max_epochs  # * self.num_batches_per_epoch
        num_labels = preds.size(-1)
        alpha = self._get_tsa_alpha()

        if self.tsa_schedule == "none":
            if mask is None:
                return preds, y_true
            else:
                return preds, y_true, mask
        if self.tsa_schedule not in ["exp", "linear", "log", "none"]:
            raise AssertionError(
                "--tsa_schedule must be in ['exp', 'linear', 'log', 'none']"
            )

        max_certainty = alpha * (1 - (1 / num_labels)) + (1 / num_labels)
        self.log("Certainty_threshold_tsa", max_certainty)

        probs = torch.softmax(preds, dim=-1)

        # Index on prob of correct class
        ix = probs[range(probs.shape[0]), y_true] < max_certainty
        if mask is not None:
            return preds[ix], y_true[ix], mask[ix]

        return preds[ix], y_true[ix]

    def get_start_node_rw(self, batch):
        row, col = batch.edge_index
        # Filter
        candidates = torch.unique(row[row >= self.doc_node_start_index])

        return candidates  # [torch.randint(0, candidates.size(0), (1,))]

    def pos_sample_similar_docs(self, batch, start_node):
        """
        Samples positive samples based on similar docs defined in graph data object
        """

        pos_nodes = [
            batch.most_similar_docs[node - self.doc_node_start_index][
                torch.randint(
                    0,
                    batch.most_similar_docs[node - self.doc_node_start_index].size(0),
                    size=(self.walks_per_node,),
                )
            ]
            for node in start_node
        ]
        pos_nodes = torch.stack(pos_nodes, dim=1).view(
            -1
        )  # torch.cat(pos_nodes, dim=0)

        # Create random walks formatted as [[start_node, pos_ex1]]
        start_node = start_node.repeat(self.walks_per_node)
        return torch.stack([start_node, pos_nodes], dim=1)

    def neg_sample_similar_docs(self, batch, start_node):
        """
        Samples negative samples based on similar docs defined in graph data object
        """

        pos_nodes = [
            batch.least_similar_docs[node - self.doc_node_start_index][
                torch.randint(
                    0,
                    batch.least_similar_docs[node - self.doc_node_start_index].size(0),
                    size=(self.walks_per_node,),
                )
            ]
            for node in start_node
        ]
        pos_nodes = torch.stack(pos_nodes, dim=1).view(
            -1
        )  # torch.cat(pos_nodes, dim=0)

        # Create random walks formatted as [[start_node, pos_ex1]]
        start_node = start_node.repeat(self.walks_per_node)
        return torch.stack([start_node, pos_nodes], dim=1)

    def unsup_step(self, batch, emb):
        # batch is complete sampled graph
        # row, col = batch.edge_index
        start_node = self.get_start_node_rw(batch)

        # Sampling
        if self.use_most_similar_docs:
            # Sample based on most overlap in vocabulary
            pos_rw = self.pos_sample_similar_docs(batch, start_node)

            neg_rw = self.neg_sample_similar_docs(batch, start_node)
        else:
            # Sample based on direct neighbour connectivity
            pos_rw = self.pos_sample(batch, start_node)
            word_nodes = pos_rw[:, 1]
            # Filter out word nodes
            pos_rw = pos_rw[:, [0, 2]]
            neg_rw = self.neg_sample(batch, start_node, pos_word_nodes=word_nodes)

        # Loss computation
        if self.use_triplet_loss:
            return self.get_triplet_loss(embeddings=emb, pos_rw=pos_rw, neg_rw=neg_rw)
        else:
            # Node2vec unsup loss
            # Replace node indices by their respective embeddings
            pos_rw = F.embedding(pos_rw, emb)
            neg_rw = F.embedding(neg_rw, emb)

            return self.unsup_loss(pos_rw, neg_rw)

    def get_triplet_loss(self, embeddings, pos_rw, neg_rw):

        triplet_indices = torch.cat([pos_rw, neg_rw[:, 1].unsqueeze(1)], dim=1)

        triplet_indices = triplet_indices.T.split(1)
        loss = self.triplet_loss(
            embeddings=embeddings,
            labels=self.triplet_dummy_labels,
            indices_tuple=triplet_indices,
        )

        return loss

    def pos_sample(self, batch, start_node):
        # Create a Random Walk from a start node to get positive samples
        row, col = batch.edge_index

        start_node = start_node.repeat(self.walks_per_node)

        rw = random_walk(row, col, start_node, self.walk_length, self.p, self.q)
        if not isinstance(rw, torch.Tensor):
            rw = rw[0]

        return rw

    def get_not_connected_nodes(self, data, start_node):

        subset, edge_index, inv, edge_mask = k_hop_subgraph(
            start_node, 1, data.edge_index
        )
        # Get node ids
        node_ids = (
            torch.ones(data.x.size(0), device=data.x.device, dtype=torch.long).cumsum(
                dim=0
            )
            - 1
        )  # Minus one as the first node is node 0
        # Create mask for non-candidate nodes
        node_mask = torch.ones_like(node_ids, dtype=torch.long)
        # Nodes in de 1-hop neighbourhood are no candidate
        node_mask[subset] = 0
        # And only document nodes are candidates
        node_mask[: self.doc_node_start_index] = 0

        unconnected_nodes = node_ids[node_mask == 1]
        return unconnected_nodes

    def neg_sample(self, batch, start_node, pos_word_nodes):
        # Sample nodes that are not connected to the start node as negative samples

        if self.neg_samples_per_word is None:
            # Find documents to which each word is not connected
            logger.info("Gathering info on documents not connected to words")
            self.neg_samples_per_word = [
                self.get_not_connected_nodes(batch, pos_word_node)
                for pos_word_node in range(self.doc_node_start_index)
            ]

        negative_nodes = [
            self.neg_samples_per_word[pos_word_node][
                torch.randint(
                    0, self.neg_samples_per_word[pos_word_node].size(0), size=(1,)
                )
            ]
            for pos_word_node in pos_word_nodes
        ]
        negative_nodes = torch.cat(negative_nodes, dim=0)

        # Create random walks formatted as [start_node, negative_ex1, negative_ex2, ..]
        start_node = start_node.repeat(self.walks_per_node)
        return torch.stack([start_node, negative_nodes], dim=1)

    @staticmethod
    def add_argparse_args(parent_parser):

        parent_parser = UnsupGNN.add_argparse_args(parent_parser)
        parser = parent_parser.add_argument_group("LitSage")

        parser.add_argument(
            "--triplet_margin",
            type=float,
            default=0.1,
            help="Initial learning rate. Set to 'auto' to use the learning rate finder.",
        )
        parser.add_argument(
            "--triplet_distance",
            default="cosine",
            help="Optimizer to use",
        )
        parser.add_argument(
            "--use_triplet_loss",
            action="store_true",
            help="Use triplet loss as unsupervised loss",
        )
        # parser.add_argument(
        #     "--use_most_similar_docs",
        #     action="store_true",
        #     help="Use unsupervised loss based on documents with most overlap in vocab",
        # )
        parser.add_argument(
            "--tsa_schedule",
            default="exp",
            help="Training Signal Annealing schedule. Choose from: ['exp', 'linear', 'log']",
        )

        return parent_parser


from typing import Optional

from torch import Tensor
from torch_sparse import SparseTensor, matmul

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor




class LitTextGNN(pl.LightningModule):
    def __init__(
        self,
        args,
        num_labels,
        num_features,
        doc_node_start_index,
        class_weights=None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(args)
        self.node_embedding_type = args.node_embedding_type
        # self.layer_dim_list = layer_dim_list
        self.hidden_dim = args.hidden_dim  # Might wanna have a penultimate dim as well
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.lr = args.lr
        self.optimizer_type = args.optimizer_type
        self.num_labels = num_labels
        self.num_features = num_features

        # This now assumes OHE of nodes
        self.layer_dim_list = (
            [self.num_features]
            + [self.hidden_dim] * (self.num_layers - 1)
            + [num_labels]
        )

        assert len(self.layer_dim_list) == (self.num_layers + 1)
        self.act_fn = args.act_fn
        self.bn = args.bn
        # self.layers = self._create_node_embd_layers()

        self.model = Sequential(
            "x, edge_index, edge_weight",
            [
                (
                    GCNConv(num_features, self.hidden_dim),
                    "x, edge_index, edge_weight -> x",
                ),
                nn.ReLU(inplace=True),
                (
                    GCNConv(self.hidden_dim, self.hidden_dim),
                    "x, edge_index, edge_weight -> x",
                ),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, self.num_labels),
            ],
        )

        self.loss = nn.CrossEntropyLoss(weight=class_weights)

    def configure_optimizers(self):
        # Start with simple Adam optimizer

        lr = 0.001 if self.hparams.lr == "auto" else float(self.hparams.lr)
        if self.optimizer_type == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        else:
            raise NotImplementedError()

        return optimizer

    def forward(self, data, mode="train"):

        x = self.model(
            data.x, data.edge_index, data.edge_attr
        )  # ,  edge_weight=data.edge_attr)

        # Only calculate the loss on the nodes corresponding to the mask
        if mode == "train":
            mask = data.train_mask
        elif mode == "val":
            mask = data.val_mask
        elif mode == "test":
            mask = data.test_mask
        else:
            assert False, f"Unknown forward mode: {mode}"

        loss = self.loss(x[mask], data.y[mask])
        acc = (x[mask].argmax(dim=-1) == data.y[mask]).sum().float() / mask.sum()
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="train")
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="val")
        self.log("val_acc", acc)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="test")
        self.log("test_acc", acc)

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("LitTextGNN")

        parser.add_argument(
            "--lr",
            type=str,
            default="auto",
            help="Initial learning rate. Set to 'auto' to use the learning rate finder.",
        )
        parser.add_argument(
            "--optimizer_type",
            default="adam",
            help="Optimizer to use",
        )
        parser.add_argument(
            "--node_embedding_type",
            default="gcn",
            help="Optimizer to use",
        )
        parser.add_argument(
            "--act_fn",
            default="prelu",
            help="Optimizer to use",
        )
        parser.add_argument(
            "--bn",
            action="store_false",
            help="Use batchnorm",
        )
        parser.add_argument(
            "--hidden_dim",
            default=64,
            type=int,
            help="Hidden dimension",
        )
        parser.add_argument(
            "--dropout",
            default=0.5,
            type=float,
            help="Dropout",
        )
        parser.add_argument(
            "--num_layers",
            default=2,
            type=int,
            help="Number of node embedding layers",
        )

        return parent_parser


class TextGNN(nn.Module):
    def __init__(
        self,
        pred_type,
        node_embd_type,
        num_layers,
        layer_dim_list,
        act,
        bn,
        num_labels,
        class_weights,
        dropout,
    ):
        super(TextGNN, self).__init__()
        self.node_embd_type = node_embd_type
        self.layer_dim_list = layer_dim_list
        self.num_layers = num_layers
        self.dropout = dropout
        if pred_type == "softmax":
            assert layer_dim_list[-1] == num_labels
        elif pred_type == "mlp":
            dims = self._calc_mlp_dims(layer_dim_list[-1], num_labels)
            self.mlp = MLP(
                layer_dim_list[-1],
                num_labels,
                num_hidden_lyr=len(dims),
                hidden_channels=dims,
                bn=False,
            )
        self.pred_type = pred_type
        assert len(layer_dim_list) == (num_layers + 1)
        self.act = act
        self.bn = bn
        self.layers = self._create_node_embd_layers()
        self.loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, pyg_graph, dataset):
        acts = [pyg_graph.x]
        for i, layer in enumerate(self.layers):
            ins = acts[-1]
            outs = layer(ins, pyg_graph)
            acts.append(outs)

        return self._loss(acts[-1], dataset)

    def _loss(self, ins, dataset):
        pred_inds = dataset.node_ids
        if self.pred_type == "softmax":
            y_preds = ins[pred_inds]
        elif self.pred_type == "mlp":
            y_preds = self.mlp(ins[pred_inds])
        else:
            raise NotImplementedError
        y_true = torch.tensor(
            dataset.label_inds[pred_inds], dtype=torch.long, device=FLAGS.device
        )
        loss = self.loss(y_preds, y_true)
        return loss, y_preds.cpu().detach().numpy()

    def _create_node_embd_layers(self):
        layers = nn.ModuleList()
        for i in range(self.num_layers):
            act = self.act if i < self.num_layers - 1 else "identity"
            layers.append(
                NodeEmbedding(
                    type=self.node_embd_type,
                    in_dim=self.layer_dim_list[i],
                    out_dim=self.layer_dim_list[i + 1],
                    act=act,
                    bn=self.bn,
                    dropout=self.dropout if i != 0 else False,
                )
            )
        return layers

    def _calc_mlp_dims(self, mlp_dim, output_dim=1):
        dim = mlp_dim
        dims = []
        while dim > output_dim:
            dim = dim // 2
            dims.append(dim)
        dims = dims[:-1]
        return dims


class NodeEmbedding(nn.Module):
    def __init__(self, type, in_dim, out_dim, act, bn, dropout, use_edge_weights=True):
        super(NodeEmbedding, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.type = type
        self.use_edge_weights = use_edge_weights

        if type == "gcn":
            self.conv = GCNConv(in_dim, out_dim)
            self.act = create_act(act, out_dim)
        elif type == "gat":
            self.conv = GATConv(in_dim, out_dim)
            self.act = create_act(act, out_dim)
        else:
            raise ValueError("Unknown node embedding layer type {}".format(type))
        self.bn = bn
        if self.bn:
            self.bn = torch.nn.BatchNorm1d(out_dim)
        self.dropout = dropout
        if dropout:
            self.dropout = torch.nn.Dropout()

    def forward(self, ins, pyg_graph):
        if self.dropout:
            ins = self.dropout(ins)
        if self.type == "gcn":
            if self.use_edge_weights:
                x = self.conv(
                    ins, pyg_graph.edge_index, edge_weight=pyg_graph.edge_attr
                )
            else:
                x = self.conv(ins, pyg_graph.edge_index)
        else:
            x = self.conv(ins, pyg_graph.edge_index)
        x = self.act(x)
        return x


class MLP(nn.Module):
    """mlp can specify number of hidden layers and hidden layer channels"""

    def __init__(
        self,
        input_dim,
        output_dim,
        activation_type="relu",
        num_hidden_lyr=2,
        hidden_channels=None,
        bn=False,
    ):
        super().__init__()
        self.out_dim = output_dim
        if not hidden_channels:
            hidden_channels = [input_dim for _ in range(num_hidden_lyr)]
        elif len(hidden_channels) != num_hidden_lyr:
            raise ValueError(
                "number of hidden layers should be the same as the lengh of hidden_channels"
            )
        self.layer_channels = [input_dim] + hidden_channels + [output_dim]
        self.activation = create_act(activation_type)
        self.layers = nn.ModuleList(
            list(
                map(
                    self.weight_init,
                    [
                        nn.Linear(self.layer_channels[i], self.layer_channels[i + 1])
                        for i in range(len(self.layer_channels) - 1)
                    ],
                )
            )
        )
        self.bn = bn
        if self.bn:
            self.bn = torch.nn.BatchNorm1d(output_dim)

    def weight_init(self, m):
        torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
        return m

    def forward(self, x):
        layer_inputs = [x]
        for layer in self.layers:
            input = layer_inputs[-1]
            if layer == self.layers[-1]:
                layer_inputs.append(layer(input))
            else:
                layer_inputs.append(self.activation(layer(input)))
        # model.store_layer_output(self, layer_inputs[-1])
        if self.bn:
            layer_inputs[-1] = self.bn(layer_inputs[-1])
        return layer_inputs[-1]


def create_act(act, num_parameters=None):
    if act == "relu":
        return nn.ReLU()
    elif act == "prelu":
        return nn.PReLU(num_parameters)
    elif act == "sigmoid":
        return nn.Sigmoid()
    elif act == "tanh":
        return nn.Tanh()
    elif act == "identity":

        class Identity(nn.Module):
            def forward(self, x):
                return x

        return Identity()
    else:
        raise ValueError("Unknown activation function {}".format(act))
