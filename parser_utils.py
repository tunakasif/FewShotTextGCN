import configargparse


def get_args():

    parser = configargparse.ArgParser(default_config_files=["parameters.ini"])

    parser.add_argument(
        "-c",
        "--my-config",
        default="parameters.ini",
        is_config_file=True,
        help="config file path",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--experiment_name",
        help="Name of the experiment",
    )
    parser.add_argument(
        "--pretrained_model",
        help="Model checkpoint to start from",
    )
    parser.add_argument(
        "--model_type",
        help="Model checkpoint to start from",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Number of epochs without improvements to stop training.",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=5,
        help="Number of seeds to run experiments for.",
    )
    parser.add_argument(
        "--preprocessed_data_dir",
        default="processed_data",
        help="Model checkpoint to start from",
    )

    args, unknown = parser.parse_known_args()

    return args, parser
