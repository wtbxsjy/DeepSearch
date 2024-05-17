import argparse


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to file with training data.",
        required=True
    )

    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to file with validation data.",
        required=True
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Directory for log storage."
    )

    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name of the experiment, if not specified, use current time.",
        required=True
    )

    parser.add_argument(
        "--n-workers",
        type=int,
        default=16,
        help="Number of dataloader workers per GPU."
    )

    parser.add_argument(
        "--batch-size", 
        type=int,
        default=128,
        help="Batch size per GPU."
    )

    parser.add_argument(
        "--n-epochs",
        type=int,
        default=25,
        help="Number of epochs to train."
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate."
    )

    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="Path to the latest checkpoint"
    )

    parser.add_argument(
        "--accum-freq",
        type=int,
        default=1,
        help="Update model every --accum-freq step "
    )
    
    parser.add_argument(
        "--augmentation",
        type=float,
        default=0.3,
        help="Data augmentation probability."
    )

    parser.add_argument(
        "--augmentation-mask",
        type=float,
        default=0.05,
        help="Data augmentation probability."
    )


    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the yaml configuration file.",
        required=True
    )

    parser.add_argument(
        "--distributed",
        action=argparse.BooleanOptionalAction,
        help="Specify for distributed training.",
        default=False
    )

    parser.add_argument(
        "--mass-anchored",
        action=argparse.BooleanOptionalAction,
        help="Specify to use mass anchored data sampler",
        default=False
    )



    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )




    args = parser.parse_args(args)

    return args 
