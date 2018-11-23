import argparse
import atexit
from typing import List

import tensorflow as tf

import const
import common.utils as utils
import factory.matting_nets as matting_nets
from datasets.data_wrapper_base import DataWrapperBase
from datasets.matting_data_wrapper import MattingDataWrapper
from factory.base import CNNModel
from helper.base import Base
from helper.trainer import MattingTrainer
from helper.trainer import TrainerBase
from metrics.base import MetricManagerBase


def train(args):
    trainer = build_trainer(args)
    trainer.train()


def build_trainer(args, trainer_cls=MattingTrainer):
    is_training = True
    session = tf.Session(config=const.TF_SESSION_CONFIG)

    # only one dataset split is assumed for training
    dataset_name = args.dataset_split_name[0]

    dataset = MattingDataWrapper(
        args,
        session,
        dataset_name,
        is_training=is_training,
    )

    images_original, masks_original, images, masks = dataset.get_input_and_output_op()

    model = eval(f"matting_nets.{args.model}")(args, dataset)
    model.build(
        images_original=images_original,
        images=images,
        masks_original=masks_original,
        masks=masks,
        is_training=is_training,
    )

    trainer = trainer_cls(
        model,
        session,
        args,
        dataset,
        dataset_name,
    )

    return trainer


def parse_arguments(arguments: List[str]=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(title="Model", description="")

    # -- * -- Common Arguments & Each Model's Arguments -- * --
    CNNModel.add_arguments(parser, default_type="matting")
    matting_nets.MattingNetModel.add_arguments(parser)
    for class_name in matting_nets._available_nets:
        subparser = subparsers.add_parser(class_name)
        subparser.add_argument("--model", default=class_name, type=str, help="DO NOT FIX ME")
        add_matting_net_arguments = eval(f"matting_nets.{class_name}.add_arguments")
        add_matting_net_arguments(subparser)

    # -- * -- Parameters & Options for MattingTrainer -- * --
    TrainerBase.add_arguments(parser)
    MattingTrainer.add_arguments(parser)
    Base.add_arguments(parser)
    DataWrapperBase.add_arguments(parser)
    MattingDataWrapper.add_arguments(parser)
    MetricManagerBase.add_arguments(parser)

    # -- Parse arguments
    args = parser.parse_args(arguments)

    # Hack!!! subparser's arguments and dynamically add it to args(Namespace)
    # it will be used for convert.py
    model_arguments = utils.get_subparser_argument_list(parser, args.model)
    args.model_arguments = model_arguments

    return args


if __name__ == "__main__":
    args = parse_arguments()
    log = utils.get_logger("MattingTrainer", None)

    utils.update_train_dir(args)

    if args.testmode:
        atexit.register(utils.exit_handler, args.train_dir)

    if args.step1_mode:
        utils.setup_step1_mode(args)

    log.info(args)
    train(args)
