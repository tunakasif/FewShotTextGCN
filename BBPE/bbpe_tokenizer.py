import sentencepiece as spm
import re
import base64
import os
import pathlib
import logging

logger = logging.getLogger(__name__)

BYTE_VOCAB_PATH = "byteVocab.txt"


class BBPETokenizer:

    # TODO: verify that implementation here creates the same output as original implementation
    def __init__(self):

        # Init bytevocab
        self.byte_vocab = self.init_byte_vocab(BYTE_VOCAB_PATH)
        # Init base 16 mapping
        b16 = {}
        for i in range(10):
            b16[str(i)] = i

        b16["A"] = 10
        b16["B"] = 11
        b16["C"] = 12
        b16["D"] = 13
        b16["E"] = 14
        b16["F"] = 15
        self.b16 = b16

        self.sentence_piece_processor = None

    def init_sentence_piece_processor(self, model_path):

        self.sentence_piece_processor = spm.SentencePieceProcessor(
            model_file=model_path
        )
        # TODO: also load word2idx?

    def init_byte_vocab(self, byte_vocab_fname):

        byte_vocab_path = os.path.join(
            pathlib.Path(__file__).parent.resolve(), byte_vocab_fname
        )
        assert os.path.isfile(byte_vocab_path)
        byte_vocab = {}
        with open(byte_vocab_path, "r", encoding="utf-8") as f:
            byte_vocab_lines = f.read().splitlines()

        assert len(byte_vocab_lines) == 512

        for line in byte_vocab_lines:
            tokens = line.strip().split("\t")
            # print(tokens[0])
            # print(tokens[1])
            byte_vocab[tokens[0]] = tokens[1]

        return byte_vocab

    def base16decode(self, s):
        result = 0
        for c in s:
            result = result * 16 + self.b16[c]
        return result

    @staticmethod
    def getPunc(context):
        #    context = context.decode("utf-8") # convert context from str to unicode
        filtrate = re.compile(
            "[^\u0020-\u002f\u003A-\u0040\u005B-\u0060\u007B-\u007E\u00A0-\u00BF\u2000-\u206f\u3000-\u303f\uff00-\uffef]"
        )  # non-Chinese unicode range
        context = filtrate.sub(r"", context)  # remove all non-Chinese characters
        #    context = context.encode("utf-8") # convert unicode back to str
        return context

    def str_to_bytes(self, line):
        """ "
        Converts a string to one out of 256 bytes, either in leading or trailing form, totalling 512 unique characters.
        Each unique character is represented by a Chinese character for compatibility with SentencePiece
        """
        result = ""
        line = line.strip()  # .split() #bytes(line.strip(), encoding="utf-8")
        lasttoken = " "
        for token in line:
            if token == " ":
                # output.write(' ')
                result += " "
                lasttoken = " "
                continue

            if lasttoken != " ":
                if len(self.getPunc(token)) > 0:
                    # output.write(" ")
                    result += " "
                    lasttoken = " "
            tk = str(base64.b16encode(token.encode("utf-8")))[2:-1]
            num = len(tk) / 2
            for i in range(int(num)):
                if lasttoken == " " and i == 0:
                    ch = str(
                        self.byte_vocab[str((self.base16decode(tk[2 * i : 2 * i + 2])))]
                    )
                else:
                    ch = str(
                        self.byte_vocab[
                            str(256 + (self.base16decode(tk[2 * i : 2 * i + 2])))
                        ]
                    )

                result += ch

            if len(self.getPunc(token)) > 0:
                # output.write(" ")
                result += " "
                lasttoken = " "
            else:
                lasttoken = token

        return result

    def docs_to_byte_dump(self, docs, out_path):
        """ "
        Takes in a corpus of documents, each document being a list of words, and converts all text to bytes and writes it to a single output file.
        This can be used to convert your corpus to a format usable by SentencePiece
        """

        logger.info(f"Creating bytes dump of {len(docs)} docs to path {out_path}")
        with open(out_path, "w", encoding="utf-8") as f:
            for doc in docs:
                # text = '\n'.join(doc)
                # f.write(self.str_to_bytes(text) + '\n')
                f.write("\n".join([self.str_to_bytes(w) for w in doc]) + "\n")

        logger.info(f"Done creating bytes dump")

    def train_sentence_piece(self, docs, dump_path, model_prefix, vocab_size):
        """ "
        Trains a SentencePiece model on the text dump found in in_path.
        For BBPE, this should be the preprocessed text, converted to 512 unique bytes (or their respective representation)

        Stores {model_prefix}.model and {model_prefix}.vocab in the current dir, which can be read in again to use for tokenization
        """
        # Create the dump
        self.docs_to_byte_dump(docs, dump_path)

        # Train the model
        spm.SentencePieceTrainer.train(
            input=dump_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
        )

    def tokenize_word(self, word):
        """ "
        Tokenize word using initiated SPM model
        """

        assert self.sentence_piece_processor is not None

        # convert to bytes?
        word_as_bytes = self.str_to_bytes(word)

        return self.sentence_piece_processor.encode(word_as_bytes)
