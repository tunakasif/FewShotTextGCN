import re
import os

# import unidecode
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from nltk.corpus import stopwords
import nltk
import numpy as np
import logging
import jieba
import fugashi

from BBPE.bbpe_tokenizer import BBPETokenizer

logger = logging.getLogger(__name__)


class TextPreprocessor:
    def __init__(self, args):

        self.debug = args.debug
        self.train_path = args.path_to_train_set
        self.test_path = args.path_to_test_set
        self.percentage_dev = args.percentage_dev

        self.language = args.language
        self.min_freq_word = args.min_freq_word
        self.num_similar_docs = args.num_similar_docs
        # For BBPE subword creation
        self.num_subwords = args.num_subwords

        # Init tokenizer
        if args.tokenizer_type == "whitespace" or args.model_type.lower() == 'transformer':
            self.tokenizer = lambda x: x.split()
        # TODO: change way of configuring
        elif args.tokenizer_type == "japanese":
            # Initialize tokenizer for e.g. Chinese and Japanese

            jp_tokenizer = fugashi.Tagger()
            self.tokenizer = lambda x: [str(word.surface) for word in jp_tokenizer(x)]
        elif args.tokenizer_type == "chinese":
            # Initialize tokenizer for e.g. Chinese and Japanese
            self.tokenizer = lambda x: [str(t[0]) for t in jieba.tokenize(x)]
        else:
            # Initialize tokenizer for e.g. Chinese and Japanese
            raise NotImplementedError()

        self.preprocess_type = args.preprocess_type

        # Placeholders
        self.word2idx = None
        self.label2idx = None
        self.train_mask = None
        self.val_mask = None
        self.test_mask = None
        self.word_counts = None

    def get_full_corpus(self):
        """
        Loads and tokenizes dataset without removin anything from it
        Used for BBPE
        """
        logger.info("Start loading full dataset without any filtering")
        # Load train set
        train_docs, train_labels = self.load_tsv(self.train_path)

        # Load test set
        test_docs, test_labels = self.load_tsv(self.test_path)

        train_docs = [self.clean_text(doc) for doc in train_docs]
        test_docs = [self.clean_text(doc) for doc in test_docs]
        # Split text
        train_docs = [self.tokenizer(doc) for doc in train_docs]
        test_docs = [self.tokenizer(doc) for doc in test_docs]

        # Filter out words based on preprocess_type and minimum frequency
        all_docs = train_docs + test_docs
        return all_docs

    def load_dataset(self):

        logger.info("Start loading dataset")
        # Load train set
        train_docs, train_labels = self.load_tsv(self.train_path)

        # Load test set
        test_docs, test_labels = self.load_tsv(self.test_path)

        # Perform cleaning
        train_docs, test_docs, word2idx = self.clean_docs(train_docs, test_docs)

        # Convert the labels
        train_labels, test_labels, label2idx = self._convert_labels(
            train_labels, test_labels
        )

        # Split train into
        train_docs, val_docs, train_labels, val_labels = self._train_test_split(
            train_docs, train_labels, self.percentage_dev
        )
        # Create masks to keep track of which document belongs to which set
        self._set_masks(train_labels, val_labels, test_labels)

        # Combine everything, since we are building one big graph
        docs = train_docs + val_docs + test_docs
        labels = np.concatenate((train_labels, val_labels, test_labels), axis=0)

        return docs, labels, word2idx, label2idx

    def get_masks(self):

        assert self.train_mask is not None
        assert self.val_mask is not None
        assert self.test_mask is not None

        return self.train_mask, self.val_mask, self.test_mask

    def _set_masks(self, train_labels, dev_labels, test_labels):
        # Create masks to distinguish docs in graph
        # Assume the first X nodes are allocated for each word, then the document nodes come
        # Document nodes are ordered as [train] + [val] + [test]
        self.train_mask = (
            [0] * len(self.word2idx)
            + [1] * len(train_labels)
            + [0] * len(dev_labels)
            + [0] * len(test_labels)
        )

        # Take as many validation samples as train samples for low-resource situations
        num_val_samples = min(len(train_labels), len(dev_labels))
        num_unused_samples = len(dev_labels) - num_val_samples
        self.val_mask = (
            [0] * len(self.word2idx)
            + [0] * len(train_labels)
            + [1] * num_val_samples
            + [0] * num_unused_samples
            + [0] * len(test_labels)
        )
        # the unlabeled samples are those not belonging to the train, val or test mask

        self.test_mask = (
            [0] * len(self.word2idx)
            + [0] * len(train_labels)
            + [0] * len(dev_labels)
            + [1] * len(test_labels)
        )

    def _convert_labels(self, train_labels, test_labels):
        # Convert labels to indices
        if not self.debug:
            assert set(train_labels) == set(
                test_labels
            ), "Mismatch in labels among sets"

        label_encoder = LabelEncoder()
        label_encoder.fit(train_labels)

        # Create a mapping
        label2idx = {
            label: label_encoder.transform([label])[0]
            for label in label_encoder.classes_
        }
        self.label2idx = label2idx
        # Convert labels
        train_labels = label_encoder.transform(train_labels)
        test_labels = label_encoder.transform(test_labels)

        return train_labels, test_labels, label2idx

    @staticmethod
    def _train_test_split(docs, labels, percentage_test):
        # Apply a stratified train/test split
        train_docs, test_docs, train_labels, test_labels = train_test_split(
            docs, labels, test_size=percentage_test, random_state=42
        )

        return train_docs, test_docs, train_labels, test_labels

    def clean_text(self, text):

        if self.preprocess_type == "textgnn":
            return clean_str(text)
        else:
            # No specific cleaning
            return text

    def remove_words(self, docs, word_counter):

        words_to_remove = [
            w for w, freq in word_counter.items() if freq < self.min_freq_word
        ]

        if self.preprocess_type == "textgnn":
            # log
            nltk.download("stopwords")
            stop_words = list(set(stopwords.words(self.language)))
            words_to_remove += stop_words

        docs = [[w for w in doc if w not in words_to_remove] for doc in docs]

        return docs

    def get_docs_with_most_overlapping_vocab(self, docs):
        logger.info(
            f"Finding documents which share the most words with the highest IDF for {len(docs)} docs"
        )

        doc_words = [set(doc) for doc in docs]

        min_overlap = []
        max_overlap = []

        for doc1 in doc_words:
            overlap = [
                float(len(doc1 & doc2)) / float(len(doc1 | doc2))
                for doc2 in doc_words
                if float(len(doc1 & doc2)) / float(len(doc1 | doc2)) != 1
            ]

            # Find the most and the least overlapping
            max_overlap_ix = np.argpartition(overlap, -self.num_similar_docs)[
                -self.num_similar_docs :
            ]
            min_overlap_ix = np.argpartition(
                [-1 * x for x in overlap], -self.num_similar_docs
            )[-self.num_similar_docs :]

            # Store per doc
            min_overlap.append(min_overlap_ix)
            max_overlap.append(max_overlap_ix)

        # Post-process: convert to doc id
        min_overlap = [[len(self.word2idx) + ix for ix in doc] for doc in min_overlap]
        max_overlap = [[len(self.word2idx) + ix for ix in doc] for doc in max_overlap]

        return max_overlap, min_overlap

    def clean_docs(self, train_docs, test_docs):
        # TODO: when/where to apply BBPE?

        # Clean text
        train_docs = [self.clean_text(doc) for doc in train_docs]
        test_docs = [self.clean_text(doc) for doc in test_docs]
        # Split text
        train_docs = [self.tokenizer(doc) for doc in train_docs]
        test_docs = [self.tokenizer(doc) for doc in test_docs]

        # Filter out words based on preprocess_type and minimum frequency
        all_docs = train_docs + test_docs
        # post-process: remove remaining whitespaces (for Japanese and Chinese)
        # all_docs = [
        #     [''.join(word.split()) for word in doc if word != ''] for doc in all_docs
        # ]

        # TODO: how to remove the whitespace from chinese tokenized text?
        # regex = re.compile(
        #     "((?<![a-zA-Z]{2})(?<=[a-zA-Z]{1})\s+(?=[a-zA-Z]\s|.$)|(?<=[\u4e00-\u9fff]{1})\s+)",
        #     re.UNICODE,
        # )
        # # s.str.replace(regex, "")
        #
        # all_docs = [
        #     [re.sub(regex, "", word) for word in doc if word != ''] for doc in all_docs
        # ]

        # Filter out words not occurring enough before creating word mapping
        if self.min_freq_word > 1:
            word_counts = Counter([word for doc in all_docs for word in doc])
            candidate_words = Counter(
                {k: c for k, c in word_counts.items() if c >= self.min_freq_word}
            )
            word2idx = {
                word: ix for ix, (word, count) in enumerate(candidate_words.items())
            }
            # Remove words not occurring enough
            train_docs = self.remove_words(train_docs, word_counts)
            test_docs = self.remove_words(test_docs, word_counts)
        else:
            # don't bother to count
            all_words = list(set([w for doc in all_docs for w in doc]))
            word2idx = {w: ix for ix, w in enumerate(all_words)}

        self.word2idx = word2idx

        return train_docs, test_docs, word2idx

    def load_tsv(self, filename):
        """
        Loads a tab separated file with a single label
        """
        nrows = None
        if self.debug:
            logger.info("Only loading first 50 documents for debugging purposes")
            nrows = 50

        df = pd.read_csv(
            filename, delimiter="\t", encoding="utf-8", header=None, nrows=nrows
        )
        assert (
            df.shape[1] == 2
        ), f"Expected two columns, but got {df.shape[1]} for file {filename}"

        labels = df.loc[:, 0].values
        docs = df.loc[:, 1].values

        return docs, labels

    def get_bbpe_info(self, docs, word2idx):
        """ """
        self.bbpe_tokenizer = BBPETokenizer()

        model_name = "spm_model_test"
        self.bbpe_tokenizer.train_sentence_piece(
            docs, "bytes_dump.txt", model_name, self.num_subwords
        )

        self.bbpe_tokenizer.init_sentence_piece_processor(
            model_path=model_name + ".model"
        )

        wordid2bbpe = {
            ix: self.bbpe_tokenizer.tokenize_word(word) for word, ix in word2idx.items()
        }

        return wordid2bbpe

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("TextPreprocessor")

        parser.add_argument(
            "--min_freq_word",
            type=int,
            default=5,
            help="Minimum frequency of a word for it to get its own word embedding.",
        )
        parser.add_argument(
            "--num_similar_docs",
            type=int,
            default=20,
            help="Number of docs with most overlapping vocab to use for unsupervised loss",
        )
        parser.add_argument(
            "--use_bbpe",
            action="store_true",
            help="Model checkpoint to start from",
        )
        parser.add_argument("--tokenizer_type", default="whitespace")
        parser.add_argument("--language", default="english")
        parser.add_argument(
            "--preprocess_type",
            default="textgcn",
            help="Kind of text preprocessing to use",
        )
        parser.add_argument(
            "--use_most_similar_docs",
            action="store_true",
            help="Use unsupervised loss based on documents with most overlap in vocab",
        )
        parser.add_argument(
            "--path_to_word2idx",
            default="word2idx.json",
            help="Path to the to the *.json word mapping file.",
        )
        parser.add_argument(
            "--path_to_train_set",
            default="datasets/reuters_train.tsv",
            help="Path to the dataset.",
        )
        parser.add_argument(
            "--path_to_test_set",
            default="datasets/reuters_test.tsv",
            help="Path to the dataset.",
        )
        parser.add_argument(
            "--percentage_dev",
            type=float,
            default=0.25,
            help="Percentage of docs to use for validation.",
        )
        parser.add_argument(
            "--num_subwords",
            type=int,
            default=512,
            help="Number of unique byte-combinations to build vocab from",
        )
        # parser.add_argument(
        #     "--use_most_similar_docs",
        #     action="store_true",
        #     help="Use unsupervised loss based on documents with most overlap in vocab",
        # )

        return parent_parser


# Clean Documents


def make_ascii(s):
    """
    Replaces chars like "ą/ę/ś/ć" with "a/e/s/c".
    This might be bad for some languages but makes it simpler for now
    """
    unaccented_string = unidecode.unidecode(s)

    return unaccented_string


def clean_str(s):
    # Replace all characters not common in English language
    # This might have to be updated later for multilingual support
    s = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", s)
    # Pull apart contractions for better matching of words with pretrained embeddings
    s = re.sub(r"\'s", " 's", s)
    s = re.sub(r"\'ve", " 've", s)
    s = re.sub(r"n\'t", " n't", s)
    s = re.sub(r"\'re", " 're", s)
    s = re.sub(r"\'d", " 'd", s)
    s = re.sub(r"\'ll", " 'll", s)
    # Add extra space around punctuation marks
    s = re.sub(r",", " , ", s)
    s = re.sub(r"!", " ! ", s)
    s = re.sub(r"\(", " \( ", s)
    s = re.sub(r"\)", " \) ", s)
    s = re.sub(r"\?", " \? ", s)
    # Normalize multiple spaces to a single on
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip().lower()  # .split()


def clean_doc(doc):
    return clean_str(make_ascii(doc))


def clean_docs(docs):
    for i in range(len(docs)):
        docs[i] = clean_doc(docs[i])
    return docs


# Fix Labels

# possible label preprocessing

# All together


def clean_data(path):
    """
    Combines all dataprep functions,
        loads data
        cleans the strings
        makes labels numeric
    returns
        docs : list of lists of strings
        labels : list of integers
    """

    docs, labels = load_file(path)

    docs = clean_docs(docs)

    print("[dataprep] Found and cleaned %d documents" % len(docs))

    return docs, labels
