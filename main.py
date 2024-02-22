from copy import deepcopy
import os
import configargparse
import time
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger

# import mlflow.pytorch
# from mlflow.tracking.context import registry as context_registry
# from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME
import logging
import logging.config
import numpy as np
import pickle
import torch
from parser_utils import get_args
from preprocess_data import TextPreprocessor
from BBPE.bbpe_tokenizer import BBPETokenizer
from dataloader import TextGCNDataset, GraphSageTextDataset, CoraDataModule
from dataloader_transformers import TransformersTextClassDatamodule

# from models import LitTextGNN, LitSage, CustomDocGraphGNN, SimpleTextGCN
from models import LitTextGNN, CustomDocGraphGNN
from models_transformers import LitTransformer

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# logging.config.fileConfig('logging.config')  # TODO: stream logs to file?
logger = logging.getLogger(__name__)

DOCS_KEY = "docs"
LABELS_KEY = "labels"
WORD2IDX_KEY = "word2idx"
LABEL2IDX_KEY = "label2idx"
TRAIN_MASK_KEY = "train_mask"
VAL_MASK_KEY = "val_mask"
TEST_MASK_KEY = "test_mask"
MOST_SIMILAR_DOCS_KEY = "most_similar_docs"
LEAST_SIMILAR_DOCS_KEY = "least_similar_docs"
EXPERIMENTS_BASE_FOLDER = "experiments"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ##################################################################
    # Collect args per module
    ##################################################################

    # Main args
    args, parser = get_args()

    # Text preprocessing args
    parser = TextPreprocessor.add_argparse_args(parser)

    # Training args
    parser = pl.Trainer.add_argparse_args(parser)

    # Choose right base classes for datamodule and model
    if args.model_type.lower() == "gcn":
        model_type = LitTextGNN
        dataset_type = TextGCNDataset
        use_word_word_edges = True
    # elif args.model_type.lower() == "graphsage":
    #     model_type = LitSage
    #     dataset_type = GraphSageTextDataset
    #     use_word_word_edges = True
    elif args.model_type.lower() == "custom":
        model_type = CustomDocGraphGNN
        dataset_type = TextGCNDataset
        use_word_word_edges = False  # args.use_unsup_loss #
        # use_word_word_edges = True #args.use_unsup_loss #
    # elif args.model_type.lower() == "simple":
    #     model_type = SimpleTextGCN
    #     dataset_type = TextGCNDataset
    #     use_word_word_edges = False
    elif args.model_type.lower() == "transformer":
        model_type = LitTransformer
        dataset_type = TransformersTextClassDatamodule
        use_word_word_edges = None
    else:
        raise NotImplementedError()

    # Model args
    parser = model_type.add_argparse_args(parser)
    # Dataset args
    parser = dataset_type.add_argparse_args(parser)

    # Workaround to parse args from config file twice
    _actions = deepcopy(parser._actions)
    _option_string_actions = deepcopy(parser._option_string_actions)
    parser = configargparse.ArgParser(default_config_files=[args.my_config])
    parser._actions = _actions
    parser._option_string_actions = _option_string_actions
    # End workaround

    args, unknown = parser.parse_known_args()

    logger.info(f"parsed args: {args}")
    if len(unknown) > 0:
        logger.info(f"Unknown args: {unknown}")

    ##################################################################
    # Read in documents
    ##################################################################
    # Initiate
    text_preprocessor = TextPreprocessor(args)
    os.makedirs(args.preprocessed_data_dir, exist_ok=True)

    preprocessed_data_path = os.path.join(
        args.preprocessed_data_dir,
        f"{os.path.splitext(os.path.split(args.path_to_train_set)[-1])[0]}_min_freq_word_{args.min_freq_word}_num_similar_docs_{args.num_similar_docs}_debug_{args.debug}_percentage_dev_{args.percentage_dev}.pkl",
    )
    if os.path.isfile(preprocessed_data_path):
        # Data has been preprocessed already, just load it
        logger.info(f"Loading existing data from {preprocessed_data_path}")
        with open(preprocessed_data_path, "rb") as f:
            preprocessed_data = pickle.load(f)

        docs = preprocessed_data[DOCS_KEY]
        labels = preprocessed_data[LABELS_KEY]
        word2idx = preprocessed_data[WORD2IDX_KEY]
        label2idx = preprocessed_data[LABEL2IDX_KEY]
        train_mask = preprocessed_data[TRAIN_MASK_KEY]
        val_mask = preprocessed_data[VAL_MASK_KEY]
        test_mask = preprocessed_data[TEST_MASK_KEY]
        most_similar_docs = preprocessed_data[MOST_SIMILAR_DOCS_KEY]
        least_similar_docs = preprocessed_data[LEAST_SIMILAR_DOCS_KEY]

    else:
        logger.info("Preprocessing data from scratch")

        # Load data
        docs, labels, word2idx, label2idx = text_preprocessor.load_dataset()
        train_mask, val_mask, test_mask = text_preprocessor.get_masks()
        if args.use_most_similar_docs:
            (
                most_similar_docs,
                least_similar_docs,
            ) = text_preprocessor.get_docs_with_most_overlapping_vocab(docs)
        else:
            # most_similar_docs, least_similar_docs = None, None
            most_similar_docs = [[0] * args.num_similar_docs] * len(docs)
            least_similar_docs = [[0] * args.num_similar_docs] * len(docs)

        # For storing
        preprocessed_data = {
            DOCS_KEY: docs,
            LABELS_KEY: labels,
            WORD2IDX_KEY: word2idx,
            LABEL2IDX_KEY: label2idx,
            TRAIN_MASK_KEY: train_mask,
            VAL_MASK_KEY: val_mask,
            TEST_MASK_KEY: test_mask,
            MOST_SIMILAR_DOCS_KEY: most_similar_docs,
            LEAST_SIMILAR_DOCS_KEY: least_similar_docs,
        }

        with open(preprocessed_data_path, "wb") as f:
            pickle.dump(preprocessed_data, f)

    bbpe_info = None
    if args.use_bbpe:
        docs_unfiltered = text_preprocessor.get_full_corpus()
        bbpe_info = text_preprocessor.get_bbpe_info(docs_unfiltered, word2idx)

    ##################################################################
    # Create graph dataset
    ##################################################################

    datamodule = dataset_type(
        args=args,
        docs=docs,
        labels=labels,
        word2idx=word2idx,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        idx2bbpe=bbpe_info,
        use_word_word_edges=use_word_word_edges,
        most_similar_docs=most_similar_docs,
        least_similar_docs=least_similar_docs,
    )

    # datamodule = CoraDataModule()

    ##################################################################
    # Init model
    ##################################################################
    doc_node_start_index = len(word2idx)
    num_features = len(word2idx) + len(docs)
    # num_features = 1433
    num_labels = len(label2idx)
    # num_labels = 7

    experiment_name = deepcopy(args.experiment_name)
    for seed in range(args.num_seeds):
        curr_seed = 42 + seed
        torch.cuda.empty_cache()

        wait_time = 15
        logger.warning(f"Waiting for {wait_time} seconds")
        time.sleep(wait_time)
        args.experiment_name = os.path.join(
            EXPERIMENTS_BASE_FOLDER, f"{experiment_name}_seed_{curr_seed}"
        )

        model = model_type(
            args=args,
            num_labels=num_labels,
            num_features=num_features,
            doc_node_start_index=doc_node_start_index,
            num_doc_nodes=len(docs),
            num_subwords=args.num_subwords,
        )



        ##################################################################
        # Train and eval
        ##################################################################

        pl.seed_everything(curr_seed)

        # Create a PyTorch Lightning trainer with the generation callback
        root_dir = args.experiment_name
        model_save_dir = os.path.join(root_dir, "models")
        os.makedirs(root_dir, exist_ok=True)

        # Setup callbacks
        early_stopping = EarlyStopping(
            monitor="val_loss", mode="min", verbose=False, patience=args.patience
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=model_save_dir,
            save_top_k=1,
            verbose=False,
            monitor="val_loss",  # TODO: Accuracy instead?
            mode="min",
            # auto_insert_metric_name=True
            # prefix="",
        )
        lr_logger = LearningRateMonitor()

        # Initiate loggers
        tb_logger = TensorBoardLogger(
            save_dir="runs", version=1, name=args.experiment_name
        )

        callbacks = [lr_logger, early_stopping, checkpoint_callback]

        # Initiate trainer
        trainer = pl.Trainer.from_argparse_args(
            args,
            callbacks=callbacks,
            # checkpoint_callback=checkpoint_callback,
            logger=[tb_logger],  # mlf_logger],
            # default_root_dir=model_save_dir,
            resume_from_checkpoint=args.pretrained_model,
            auto_lr_find=args.lr == "auto",
            benchmark=True,  # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
            # deterministic=True
            # profiler="advanced"
        )

        # Start the actual training
        # with mlflow.start_run() as run:
        trainer.fit(model, datamodule)

        # Test best model on the test set
        test_result = trainer.test(
            model, datamodule=datamodule,  verbose=False, ckpt_path="best" #test_dataloaders=datamodule,
        )
        logger.info(f"Test result {test_result}")
        print(test_result)

        del trainer, model
