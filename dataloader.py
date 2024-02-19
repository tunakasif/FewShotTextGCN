import abc
import numpy as np
import random
import torch

from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from collections import defaultdict, Counter
from math import log
from torch_cluster import random_walk
from torch_geometric.data import Data, DataLoader
from torch_geometric.loader import NeighborLoader
from torch_geometric.loader.utils import filter_data
from torch_geometric.datasets import Planetoid
from torch_sparse import SparseTensor
import pytorch_lightning as pl
import scipy.sparse as sp
import logging

logger = logging.getLogger(__name__)


class CoraDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.data = None

    def setup(self, stage=None):

        self.data = Planetoid("path", name="Cora")[0]

    def train_dataloader(self):

        return NeighborLoader(
            self.data,
            # Sample 30 neighbors for each node for 2 iterations
            num_neighbors=[30] * 2,
            # Use a batch size of 128 for sampling training nodes
            batch_size=128,
            input_nodes=self.data.train_mask,
        )

    def val_dataloader(self):
        return NeighborLoader(
            self.data,
            # Sample 30 neighbors for each node for 2 iterations
            num_neighbors=[30] * 2,
            # Use a batch size of 128 for sampling training nodes
            batch_size=128,
            input_nodes=self.data.val_mask,
        )

    def test_dataloader(self):
        return NeighborLoader(
            self.data,
            # Sample 30 neighbors for each node for 2 iterations
            num_neighbors=[30] * 2,
            # Use a batch size of 128 for sampling training nodes
            batch_size=128,
            input_nodes=self.data.test_mask,
        )


class DocumentGraphDataset(pl.LightningDataModule):

    """ """

    def __init__(
        self,
        args,
        docs,
        labels,
        word2idx,
        train_mask,
        val_mask,
        test_mask,
        use_word_word_edges,
        most_similar_docs,
        least_similar_docs,
        idx2bbpe=None,
        loaded_dict=None,
    ):

        super().__init__()
        if loaded_dict is not None:  # restore from content loaded from disk
            self.__dict__ = loaded_dict
            return

        assert len(docs) == len(
            labels
        ), "Mismatch in number of documents and number of labels"

        logger.info(
            f"Start creating graph dataset with {len(word2idx)} unique words and {len(docs)} documents."
        )
        logger.info(
            f"Number of documents per set - train: {np.sum(train_mask)} val: {np.sum(val_mask)} test {np.sum(test_mask)}"
        )
        self.args = args
        self.docs = docs
        self.labels = labels
        self.word2idx = word2idx
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.use_word_word_edges = use_word_word_edges
        self.most_similar_docs = most_similar_docs
        self.least_similar_doc = least_similar_docs
        self.idx2bbpe = idx2bbpe

    def setup(self, stage=None):

        # Create adjacency matrix
        self.adj = self.create_graph()

        # Init the node features
        self.node_features = self.init_node_features()

        # Create final graph obj
        adj = self.adj
        A = adj.tocoo()
        row = torch.from_numpy(A.row).to(torch.long)
        col = torch.from_numpy(A.col).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)
        edge_weight = torch.from_numpy(A.data).to(torch.float)
        if type(self.node_features) is not torch.Tensor:
            gx = torch.tensor(self.node_feats, dtype=torch.float32)
        else:
            gx = self.node_features

        # Create dummy labels for word nodes
        y = np.zeros(len(self.word2idx) + len(self.labels))
        y[-len(self.labels) :] = self.labels
        self.graph = Data(
            x=gx,
            y=torch.LongTensor(y),
            edge_index=edge_index,
            edge_attr=edge_weight,
            train_mask=torch.BoolTensor(self.train_mask),
            val_mask=torch.BoolTensor(self.val_mask),
            test_mask=torch.BoolTensor(self.test_mask),
            least_similar_docs=torch.LongTensor(self.least_similar_doc),
            most_similar_docs=torch.LongTensor(self.most_similar_docs),
        )

    def save(self):
        # TODO
        raise NotImplementedError()

    def load(self):
        # TODO
        raise NotImplementedError()

    @abc.abstractmethod
    def create_graph(self):
        pass

    @abc.abstractmethod
    def init_node_features(self):
        pass

    def __getitem__(self, index):
        # idea: make actual graph part of PL module, only load indexes of nodes and corresponding labels
        return self.graph

    def __len__(self):
        return 1

    def train_dataloader(self):
        return DataLoader(self, batch_size=1)

    def val_dataloader(self):
        return DataLoader(self, batch_size=1)

    def test_dataloader(self):
        return DataLoader(self, batch_size=1)


def count_co_occurences(
    doc, window_size, return_count=False, co_occurences=defaultdict(int)
):
    window_count = 0
    for i, w in enumerate(doc):
        for j in range(i + 1, min(i + window_size + 1, len(doc))):
            window_count += 1
            if (doc[i], doc[j]) in co_occurences:
                co_occurences[
                    (doc[i], doc[j])
                ] += 1  # Could add weighting based on distance
            else:
                co_occurences[(doc[j], doc[i])] += 1

    if return_count:
        return co_occurences, window_count
    return co_occurences


class TextGCNDataset(DocumentGraphDataset, pl.LightningDataModule):
    def __init__(
        self,
        args,
        docs,
        labels,
        word2idx,
        train_mask,
        val_mask,
        test_mask,
        most_similar_docs,
        least_similar_docs,
        use_word_word_edges,
        idx2bbpe=None,
        loaded_dict=None,
    ):
        super().__init__(
            args=args,
            docs=docs,
            labels=labels,
            word2idx=word2idx,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            use_word_word_edges=use_word_word_edges,
            most_similar_docs=most_similar_docs,
            least_similar_docs=least_similar_docs,
            idx2bbpe=idx2bbpe,
            loaded_dict=loaded_dict,
        )
        self.use_word_word_edges = use_word_word_edges
        self.window_size = args.window_size

    def _get_word_edges(self, docs, word2idx, window_size):

        total_window_count = 0
        co_occurences = defaultdict(int)
        for doc in docs:
            co_occurences, window_count = count_co_occurences(
                doc,
                window_size=window_size,
                return_count=True,
                co_occurences=co_occurences,
            )
            total_window_count += window_count
        row = []
        col = []
        weight = []

        # pmi as weights
        num_docs = len(docs)
        num_window = total_window_count
        vocab = list(word2idx.keys())
        num_words = len(vocab)

        if self.use_word_word_edges:
            word_freq = Counter([word for doc in docs for word in doc])

            for ((word_a, word_b), count) in co_occurences.items():
                # pmi = log ( p(x,y) / (p(x)p(y)) )
                pmi = log(
                    (1.0 * count / total_window_count)
                    / (
                        (1.0 * word_freq[word_a] * word_freq[word_b])
                        / (total_window_count ** 2)
                    )
                )  # TODO: not tested if correct

                if pmi <= 0:
                    continue

                row.append(word2idx[word_a])
                col.append(word2idx[word_b])
                weight.append(pmi)

        # frequency of document word pair
        doc_word_freq = defaultdict(int)
        for i, words in enumerate(docs):
            # words = doc_words.split()
            for word in words:
                word_id = word2idx[word]
                doc_word_str = (i, word_id)
                doc_word_freq[doc_word_str] += 1

        # Get word-doc frequency
        word_doc_freq = self._get_word_doc_freq(docs)

        for i, words in enumerate(docs):
            # words = doc_words.split()
            doc_word_set = set()
            for word in words:
                if word in doc_word_set:
                    continue
                word_id = word2idx[word]
                freq = doc_word_freq[(i, word_id)]  # / len(words)
                row.append(num_words + i)
                col.append(word_id)
                idf = log(1.0 * num_docs / word_doc_freq[vocab[word_id]])
                weight.append(freq * idf)
                doc_word_set.add(word)

        number_nodes = num_docs + num_words
        adj_mat = sp.csr_matrix(
            (weight, (row, col)), shape=(number_nodes, number_nodes)
        )
        adj = (
            adj_mat
            + adj_mat.T.multiply(adj_mat.T > adj_mat)
            - adj_mat.multiply(adj_mat.T > adj_mat)
        )
        return adj

    def _get_word_doc_freq(self, docs):
        # build all docs that a word is contained in
        words_in_docs = defaultdict(set)
        for i, words in enumerate(docs):
            # words = doc_words.split()
            for word in words:
                words_in_docs[word].add(i)

        word_doc_freq = {}
        for word, doc_list in words_in_docs.items():
            word_doc_freq[word] = len(doc_list)

        return word_doc_freq

    def init_node_features(self):

        if self.idx2bbpe is None:
            # One-hot-encoding
            return self.init_node_features_ohe()
        else:
            return self.init_node_features_bbpe()

    def init_node_features_ohe(self):
        num_nodes = self.adj.shape[0]
        identity = sp.identity(num_nodes)
        ind0, ind1, values = sp.find(identity)
        inds = np.stack((ind0, ind1), axis=0)
        node_feats = torch.sparse_coo_tensor(inds, values, dtype=torch.float)

        return node_feats

    def init_node_features_bbpe(self):  # from bbpe
        """
        Embeds words as all its BBPE parts.
        Docs all have the same embedding to start with
        """

        num_pad_emb = 1
        pad_ix = 0
        num_subwords = max(
            [subw for i, w in self.idx2bbpe.items() for subw in w]
        )  # len(self.idx2bbpe)
        num_doc_nodes = len(self.docs)

        # Unique embedding per doc:
        doc_node_features = [
            torch.LongTensor([doc_id])
            for doc_id in range(
                num_subwords + num_pad_emb, num_pad_emb + num_subwords + num_doc_nodes
            )
        ]

        # Same embedding per doc:
        # doc_node_features = [torch.LongTensor([num_subwords+num_pad_emb]) for doc_id in range(num_subwords+num_pad_emb, num_pad_emb+num_subwords+num_doc_nodes)]

        word_node_features = [
            torch.LongTensor(self.idx2bbpe[ix]) for ix in range(len(self.word2idx))
        ]

        num_subwords_per_word = [w.size(0) for w in word_node_features]
        logger.info(
            f"Words have an average of {np.round(np.mean(num_subwords_per_word))} subwords per word"
        )

        node_features = pad_sequence(
            word_node_features + doc_node_features,
            batch_first=True,
            padding_value=pad_ix,
        )

        return node_features

    def create_graph(self):

        adj = self._get_word_edges(
            docs=self.docs, word2idx=self.word2idx, window_size=self.window_size
        )
        return adj

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("TextGCNDataset")

        parser.add_argument(
            "--window_size",
            type=int,
            default=20,
            help="Window size for words to compute PMI.",
        )

        return parent_parser


class GraphSageTextDataset(TextGCNDataset):
    def __init__(
        self,
        args,
        docs,
        labels,
        word2idx,
        train_mask,
        val_mask,
        test_mask,
        use_word_word_edges,
        most_similar_docs,
        least_similar_docs,
        loaded_dict=None,
    ):
        super().__init__(
            args=args,
            docs=docs,
            labels=labels,
            word2idx=word2idx,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            use_word_word_edges=use_word_word_edges,
            most_similar_docs=most_similar_docs,
            least_similar_docs=least_similar_docs,
            loaded_dict=loaded_dict,
        )

        self.use_unsup_loss = args.use_unsup_loss
        self.batch_size = args.batch_size
        self.num_neighbors_hop1 = args.num_neighbors_hop1
        self.num_neighbors_hop2 = args.num_neighbors_hop2

    def init_node_features(self):
        # Create a dense identity matrix
        num_nodes = self.adj.shape[0]
        identity = np.identity(num_nodes)
        node_feats = torch.FloatTensor(identity)

        return node_feats

    def train_dataloader(self):
        return NeighborLoader(
            self.graph,
            num_neighbors=[self.num_neighbors_hop1, self.num_neighbors_hop2],
            batch_size=self.batch_size,
            input_nodes=self.graph.train_mask,
        )

    def val_dataloader(self):
        return NeighborLoader(
            self.graph,
            num_neighbors=[self.num_neighbors_hop1, self.num_neighbors_hop2],
            batch_size=self.batch_size,
            input_nodes=self.graph.val_mask,
        )

    def test_dataloader(self):
        return NeighborLoader(
            self.graph,
            num_neighbors=[self.num_neighbors_hop1, self.num_neighbors_hop2],
            batch_size=self.batch_size,
            input_nodes=self.graph.test_mask,
        )

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("GraphSageTextDataset")

        parser.add_argument(
            "--window_size",
            type=int,
            default=20,
            help="Window size for words to compute PMI.",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=20,
            help="Window size for words to compute PMI.",
        )
        parser.add_argument(
            "--num_neighbors_hop1",
            type=int,
            default=30,
            help="Window size for words to compute PMI.",
        )
        parser.add_argument(
            "--num_neighbors_hop2",
            type=int,
            default=15,
            help="Window size for words to compute PMI.",
        )

        return parent_parser
