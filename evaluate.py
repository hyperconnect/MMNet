import argparse
from typing import List

import tensorflow as tf

from common.tf_utils import ckpt_iterator
import common.utils as utils
import const
from datasets.data_wrapper_base import DataWrapperBase
from datasets.matting_data_wrapper import MattingDataWrapper
from factory.base import CNNModel
import factory.matting_nets as matting_nets
from helper.base import Base
from helper.evaluator import Evaluator
from helper.evaluator import MattingEvaluator
from metrics.base import MetricManagerBase


def evaluate(args):
    evaluator = build_evaluator(args)
    log = utils.get_logger("EvaluateMatting")
    dataset_names = args.dataset_split_name

    if args.inference:
        for dataset_name in dataset_names:
            evaluator[dataset_name].inference(args.checkpoint_path)
    else:
        if args.valid_type == "once":
            for dataset_name in dataset_names:
                evaluator[dataset_name].evaluate_once(args.checkpoint_path)
        elif args.valid_type == "loop":
            current_evaluator = evaluator[dataset_names[0]]
            log.info(f"Start Loop: watching {current_evaluator.watch_path}")

            kwargs = {
                "min_interval_secs": 0,
                "timeout": None,
                "timeout_fn": None,
                "logger": log,
            }
            for ckpt_path in ckpt_iterator(current_evaluator.watch_path, **kwargs):
                log.info(f"[watch] {ckpt_path}")

                for dataset_name in dataset_names:
                    evaluator[dataset_name].evaluate_once(ckpt_path)
        else:
            raise ValueError(f"Undefined valid_type: {args.valid_type}")


def build_evaluator(args, evaluator_cls=MattingEvaluator):
    session = tf.Session(config=const.TF_SESSION_CONFIG)
    dataset_names = args.dataset_split_name

    dataset = MattingDataWrapper(
        args,
        session,
        dataset_names[0],
        is_training=False,
    )
    images_original, masks_original, images, masks = dataset.get_input_and_output_op()
    model = eval("matting_nets.{}".format(args.model))(args, dataset)
    model.build(
        images_original=images_original,
        images=images,
        masks_original=masks_original,
        masks=masks,
        is_training=False,
    )

    evaluator = {
        dataset_names[0]: evaluator_cls(
            model,
            session,
            args,
            dataset,
            dataset_names[0],
        )
    }
    for dataset_name in dataset_names[1:]:
        assert False, "Evaluation of multiple dataset splits does not work."
        dataset = MattingDataWrapper(
            args,
            session,
            dataset_name,
            is_training=False,
        )

        evaluator[dataset_name] = evaluator_cls(
            model,
            session,
            args,
            dataset,
            dataset_name,
        )

    return evaluator


def parse_arguments(arguments: List[str]=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(title="Model", description="")

    # -- * -- Common Arguments & Each Model's Arguments -- * --
    CNNModel.add_arguments(parser, default_type="matting")
    matting_nets.MattingNetModel.add_arguments(parser)
    for class_name in matting_nets._available_nets:
        subparser = subparsers.add_parser(class_name)
        subparser.add_argument("--model", default=class_name, type=str, help="DO NOT FIX ME")
        add_matting_net_arguments = eval("matting_nets.{}.add_arguments".format(class_name))
        add_matting_net_arguments(subparser)

    Evaluator.add_arguments(parser)
    Base.add_arguments(parser)
    DataWrapperBase.add_arguments(parser)
    MattingDataWrapper.add_arguments(parser)
    MetricManagerBase.add_arguments(parser)

    args = parser.parse_args(arguments)

    model_arguments = utils.get_subparser_argument_list(parser, args.model)
    args.model_arguments = model_arguments

    return args


if __name__ == "__main__":
    args = parse_arguments()
    log = utils.get_logger("MattingEvaluator", None)

    log.info(args)
    evaluate(args)
