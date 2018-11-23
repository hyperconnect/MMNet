import sys
import time
from pathlib import Path
from abc import ABC
from abc import abstractmethod
from typing import Dict

import humanfriendly as hf
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from termcolor import colored

import common.tf_utils as tf_utils
import common.utils as utils
import common.lr_scheduler as lr_scheduler
import metrics.manager as metric_manager
from helper.base import MattingBase
from common.model_loader import Ckpt
from metrics.summaries import BaseSummaries
from metrics.summaries import Summaries


class TrainerBase(ABC):
    def __init__(self, model, session, args, dataset, dataset_name, name):
        self.model = model
        self.session = session
        self.args = args
        self.dataset = dataset
        self.dataset_name = dataset_name

        self.log = utils.get_logger(name)
        self.timer = utils.Timer(self.log)

        self.info_red = utils.format_log(self.log.info, "red")
        self.info_cyan = utils.format_log(self.log.info, "cyan")
        self.info_magenta = utils.format_log(self.log.info, "magenta")
        self.info_magenta_reverse = utils.format_log(self.log.info, "magenta", attrs=["reverse"])
        self.info_cyan_underline = utils.format_log(self.log.info, "cyan", attrs=["underline"])
        self.debug_red = utils.format_log(self.log.debug, "red")
        self._saver = None

        # used in `log_step_message` method
        self.last_loss = dict()

    @property
    def etc_fetch_namespace(self):
        return utils.MLNamespace(
            step_op="step_op",
            global_step="global_step",
            summary="summary",
        )

    @property
    def loss_fetch_namespace(self):
        return utils.MLNamespace(
            total_loss="total_loss",
            model_loss="model_loss",
        )

    @property
    def summary_fetch_namespace(self):
        return utils.MLNamespace(
            merged_summaries="merged_summaries",
            merged_verbose_summaries="merged_verbose_summaries",
            merged_first_n_summaries="merged_first_n_summaries",
        )

    @property
    def after_fetch_namespace(self):
        return utils.MLNamespace(
            single_step="single_step",
            single_step_per_image="single_step_per_image",
        )

    @property
    def saver(self):
        if self._saver is None:
            self._saver = tf.train.Saver(max_to_keep=self.args.max_to_keep)
        return self._saver

    @abstractmethod
    def setup_metric_manager(self):
        raise NotImplementedError

    @abstractmethod
    def setup_metric_ops(self):
        raise NotImplementedError

    @abstractmethod
    def build_non_tensor_data_from_eval_dict(self, eval_dict, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def build_evaluate_iterations(self, iters: int):
        raise NotImplementedError

    def routine_experiment_summary(self):
        self.metric_manager.summary.setup_experiment(self.args)

    def setup_essentials(self, max_to_keep=5):
        self.no_op = tf.no_op()
        self.args.checkpoint_path = tf_utils.resolve_checkpoint_path(
            self.args.checkpoint_path, self.log, is_training=True
        )
        self.train_dir_name = (Path.cwd() / Path(self.args.train_dir)).resolve()

        # We use this global step for shift boundaries for piecewise_constant learning rate
        # We cannot use global step from checkpoint file before restore from checkpoint
        # For restoring, we needs to initialize all operations including optimizer
        self.global_step_from_checkpoint = tf_utils.get_global_step_from_checkpoint(self.args.checkpoint_path)
        self.global_step = tf.Variable(self.global_step_from_checkpoint, name="global_step", trainable=False)

        self.lr_scheduler = lr_scheduler.factory(
            self.args, self.log, self.global_step_from_checkpoint, self.global_step, self.dataset
        )

    def routine_restore_and_initialize(self, checkpoint_path=None):
        """Read various loading methods for tensorflow
        - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/learning.py#L121
        """
        if checkpoint_path is None:
            checkpoint_path = self.args.checkpoint_path
        var_names_to_values = getattr(self.model, "var_names_to_values", None)
        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.local_variables_initializer())  # for metrics

        if var_names_to_values is not None:
            init_assign_op, init_feed_dict = slim.assign_from_values(var_names_to_values)
            # Create an initial assignment function.
            self.session.run(init_assign_op, init_feed_dict)
            self.log.info(colored("Restore from Memory(usually weights from caffe!)",
                                  "cyan", attrs=["bold", "underline"]))
        elif checkpoint_path == "" or checkpoint_path is None:
            self.log.info(colored("Initialize global / local variables", "cyan", attrs=["bold", "underline"]))
        else:
            ckpt_loader = Ckpt(
                session=self.session,
                include_scopes=self.args.checkpoint_include_scopes,
                exclude_scopes=self.args.checkpoint_exclude_scopes,
                ignore_missing_vars=self.args.ignore_missing_vars,
            )
            ckpt_loader.load(checkpoint_path)

    def routine_logging_checkpoint_path(self):
        self.log.info(colored("Watch Validation Through TensorBoard !", "yellow", attrs=["underline", "bold"]))
        self.log.info(colored("--checkpoint_path {}".format(self.train_dir_name),
                              "yellow", attrs=["underline", "bold"]))

    def build_optimizer(self, optimizer, learning_rate, momentum=None, decay=None, epsilon=None):
        kwargs = {
            "learning_rate": learning_rate
        }
        if momentum:
            kwargs["momentum"] = momentum
        if decay:
            kwargs["decay"] = decay
        if epsilon:
            kwargs["epsilon"] = epsilon

        if optimizer == "gd":
            opt = tf.train.GradientDescentOptimizer(**kwargs)
            self.log.info("Use GradientDescentOptimizer")
        elif optimizer == "adam":
            opt = tf.train.AdamOptimizer(**kwargs)
            self.log.info("Use AdamOptimizer")
        elif optimizer == "mom":
            opt = tf.train.MomentumOptimizer(**kwargs)
            self.log.info("Use MomentumOptimizer")
        elif optimizer == "rmsprop":
            opt = tf.train.RMSPropOptimizer(**kwargs)
            self.log.info("Use RMSPropOptimizer")
        else:
            self.log.error("Unknown optimizer: {}".format(optimizer))
            raise NotImplementedError
        return opt

    def build_train_op(self, total_loss, optimizer, trainable_scopes, global_step, gradient_multipliers=None):
        # If you use `slim.batch_norm`, then you should include train_op in slim.
        # https://github.com/tensorflow/tensorflow/issues/1122#issuecomment-280325584
        variables_to_train = tf_utils.get_variables_to_train(trainable_scopes, logger=self.log)

        if variables_to_train:
            train_op = slim.learning.create_train_op(
                total_loss,
                optimizer,
                global_step=global_step,
                variables_to_train=variables_to_train,
                gradient_multipliers=gradient_multipliers,
            )

            if self.args.use_ema:
                self.ema = tf.train.ExponentialMovingAverage(decay=self.args.ema_decay)

                with tf.control_dependencies([train_op]):
                    train_op = self.ema.apply(variables_to_train)
        else:
            self.log.info("Empty variables_to_train")
            train_op = tf.no_op()

        return train_op

    def build_epoch(self, step):
        return (step * self.dataset.batch_size) / self.dataset.num_samples

    def build_info_step_message(self, info: Dict, float_format, delimiter: str=" / "):
        keys = list(info.keys())
        desc = delimiter.join(keys)
        val = delimiter.join([str(float_format.format(info[k])) for k in keys])
        return desc, val

    def build_duration_step_message(self, header: Dict, delimiter: str=" / "):
        def convert_to_string(number):
            type_number = type(number)
            if type_number == int or type_number == np.int32 or type_number == np.int64:
                return str(f"{number:8d}")
            elif type_number == float or type_number == np.float64:
                return str(f"{number:3.3f}")
            else:
                raise TypeError("Unrecognized type of input number")

        keys = list(header.keys())
        header_desc = delimiter.join(keys)
        header_val = delimiter.join([convert_to_string(header[k]) for k in keys])

        return header_desc, header_val

    def log_evaluation(
        self, dataset_name, epoch_from_restore, step_from_restore, global_step, eval_scores,
    ):
        self.info_cyan_underline(
            f"[{dataset_name}-Evaluation] global_step / step_from_restore / epoch_from_restore: "
            f"{global_step:8d} / {step_from_restore:5d} / {epoch_from_restore:3.3f}"
        )
        self.metric_manager.log_metrics(global_step)

    def log_step_message(self, header, losses, speeds, comparative_loss, batch_size, is_training, tag=""):
        def get_loss_color(old_loss: float, new_loss: float):
            if old_loss < new_loss:
                return "red"
            else:
                return "green"

        def get_log_color(is_training: bool):
            if is_training:
                return {"color": "blue",
                        "attrs": ["bold"]}
            else:
                return {"color": "yellow",
                        "attrs": ["underline"]}

        self.last_loss.setdefault(tag, comparative_loss)
        loss_color = get_loss_color(self.last_loss.get(tag, 0), comparative_loss)
        self.last_loss[tag] = comparative_loss

        model_size = hf.format_size(self.model.total_params*4)
        total_params = hf.format_number(self.model.total_params)

        loss_desc, loss_val = self.build_info_step_message(losses, "{:7.4f}")
        header_desc, header_val = self.build_duration_step_message(header)
        speed_desc, speed_val = self.build_info_step_message(speeds, "{:4.0f}")

        with utils.format_text(loss_color) as fmt:
            loss_val_colored = fmt(loss_val)
            msg = (
                f"[{tag}] {header_desc}: {header_val}\t"
                f"{speed_desc}: {speed_val} ({self.args.width},{self.args.height};{batch_size})\t"
                f"{loss_desc}: {loss_val_colored} "
                f"| {model_size} {total_params}")

            with utils.format_text(**get_log_color(is_training)) as fmt:
                self.log.info(fmt(msg))

    def setup_trainer(self):
        self.setup_essentials(self.args.max_to_keep)
        self.optimizer = self.build_optimizer(self.args.optimizer,
                                              learning_rate=self.lr_scheduler.placeholder,
                                              momentum=self.args.momentum,
                                              decay=self.args.optimizer_decay,
                                              epsilon=self.args.optimizer_epsilon)
        self.train_op = self.build_train_op(total_loss=self.model.total_loss,
                                            optimizer=self.optimizer,
                                            trainable_scopes=self.args.trainable_scopes,
                                            global_step=self.global_step)
        self.routine_restore_and_initialize()

    def append_if_value_is_not_none(self, key_and_op, fetch_ops):
        if key_and_op[1] is not None:
            fetch_ops.append(key_and_op)

    def run_single_step(self, fetch_ops: Dict, feed_dict: Dict=None):
        st = time.time()
        fetch_vals = self.session.run(fetch_ops, feed_dict=feed_dict)
        step_time = (time.time() - st) * 1000
        step_time_per_image = step_time / self.dataset.batch_size

        fetch_vals[self.after_fetch_namespace.single_step] = step_time
        fetch_vals[self.after_fetch_namespace.single_step_per_image] = step_time_per_image

        return fetch_vals

    def log_summaries(self, fetch_vals):
        summary_keys = (
            set(fetch_vals.keys()) -
            set(self.loss_fetch_namespace.unordered_values()) -
            set(self.etc_fetch_namespace.unordered_values()) -
            set(self.after_fetch_namespace.unordered_values())
        )
        if len(summary_keys) > 0:
            self.info_magenta(f"Above step includes saving {summary_keys} summaries to {self.train_dir_name}")

    def run_with_logging(self, summary_op, metric_op_dict, feed_dict):
        fetch_ops = {
            self.etc_fetch_namespace.step_op: self.train_op,
            self.etc_fetch_namespace.global_step: self.global_step,
            self.loss_fetch_namespace.total_loss: self.model.total_loss,
            self.loss_fetch_namespace.model_loss: self.model.model_loss,
        }
        if summary_op is not None:
            fetch_ops.update({self.etc_fetch_namespace.summary: summary_op})
        if metric_op_dict is not None:
            fetch_ops.update(metric_op_dict)

        fetch_vals = self.run_single_step(fetch_ops=fetch_ops, feed_dict=feed_dict)

        global_step = fetch_vals[self.etc_fetch_namespace.global_step]
        step_from_restore = global_step - self.global_step_from_checkpoint
        epoch_from_restore = self.build_epoch(step_from_restore)

        self.log_step_message(
            {"GlobalStep": global_step,
             "StepFromRestore": step_from_restore,
             "EpochFromRestore": epoch_from_restore},
            {"TotalLoss": fetch_vals[self.loss_fetch_namespace.total_loss],
             "ModelLoss": fetch_vals[self.loss_fetch_namespace.model_loss]},
            {"SingleStepPerImage(ms)": fetch_vals[self.after_fetch_namespace.single_step_per_image],
             "SingleStep(ms)": fetch_vals[self.after_fetch_namespace.single_step]},
            comparative_loss=fetch_vals[self.loss_fetch_namespace.total_loss],
            batch_size=self.dataset.batch_size,
            tag=self.dataset_name,
            is_training=True
        )

        return fetch_vals, global_step, step_from_restore, epoch_from_restore

    def train(self, name: str="Training"):
        self.log.info(f"{name} started")

        global_step, step_from_restore, epoch_from_restore = 0, 0, 0
        while True:
            try:
                feed_dict = self.get_feed_dict(is_training=True)
                valid_collection_keys = []

                # collect valid collection keys
                if step_from_restore >= self.args.step_min_summaries and \
                        step_from_restore % self.args.step_save_summaries == 0:
                    valid_collection_keys.append(BaseSummaries.KEY_TYPES.DEFAULT)

                if step_from_restore % self.args.step_save_verbose_summaries == 0:
                    valid_collection_keys.append(BaseSummaries.KEY_TYPES.VERBOSE)

                if step_from_restore <= self.args.step_save_first_n_summaries:
                    valid_collection_keys.append(BaseSummaries.KEY_TYPES.FIRST_N)

                # merge it to single one
                summary_op = self.metric_manager.summary.get_merged_summaries(
                    collection_key_suffixes=valid_collection_keys,
                    is_tensor_summary=True
                )

                # send metric op
                # run it only when evaluate
                if step_from_restore % self.args.step_evaluation == 0:
                    metric_op_dict = self.metric_tf_op
                else:
                    metric_op_dict = None

                # Session.Run!
                fetch_vals, global_step, step_from_restore, epoch_from_restore = self.run_with_logging(
                    summary_op, metric_op_dict, feed_dict)
                self.log_summaries(fetch_vals)

                # Save
                if step_from_restore % self.args.step_save_checkpoint == 0:
                    with self.timer(f"save checkpoint: {self.train_dir_name}", self.info_magenta_reverse):
                        self.saver.save(self.session,
                                        str(Path(self.args.train_dir) / self.args.model),
                                        global_step=global_step)
                        if self.args.write_pbtxt:
                            tf.train.write_graph(
                                self.session.graph_def, self.args.train_dir, self.args.model + ".pbtxt"
                            )

                if step_from_restore % self.args.step_evaluation == 0:
                    self.evaluate(epoch_from_restore, step_from_restore, global_step, self.dataset_name)

                if epoch_from_restore >= self.args.max_epoch_from_restore:
                    self.info_red(f"Reached {self.args.max_epoch_from_restore} epochs from restore.")
                    break

                if step_from_restore >= self.args.max_step_from_restore:
                    self.info_red(f"Reached {self.args.max_step_from_restore} steps from restore.")
                    break

                if self.args.step1_mode:
                    break

                global_step += 1
                step_from_restore = global_step - self.global_step_from_checkpoint
                epoch_from_restore = self.build_epoch(step_from_restore)
            except tf.errors.InvalidArgumentError as e:
                utils.format_log(self.log.error, "red")(f"Invalid image is detected: {e}")
                continue

        self.log.info(f"{name} finished")

    def evaluate(
        self,
        epoch_from_restore: float,
        step_from_restore: int,
        global_step: int,
        dataset_name: str,
        iters: int=None,
    ):
        # Update learning rate based on validation loss will be implemented in new class
        # where Trainer and Evaluator will share information. Currently, `ReduceLROnPlateau`
        # would not work correctly.
        # self.lr_scheduler.update_on_start_of_evaluation()

        # calculate number of iterations
        iters = self.build_evaluate_iterations(iters)

        with self.timer(f"run_evaluation (iterations: {iters})", self.info_cyan):
            # evaluate metrics
            eval_dict = self.run_inference(global_step, iters=iters, is_training=True)

            non_tensor_data = self.build_non_tensor_data_from_eval_dict(eval_dict)

            self.metric_manager.evaluate_and_aggregate_metrics(step=global_step,
                                                               non_tensor_data=non_tensor_data,
                                                               eval_dict=eval_dict)

        self.metric_manager.write_evaluation_summaries(step=global_step,
                                                       collection_keys=[BaseSummaries.KEY_TYPES.DEFAULT])

        self.log_evaluation(dataset_name, epoch_from_restore, step_from_restore, global_step, eval_scores=None)


    @staticmethod
    def add_arguments(parser, name: str="TrainerBase"):
        g_optimize = parser.add_argument_group(f"({name}) Optimizer Arguments")
        g_optimize.add_argument("--optimizer", default="adam", type=str,
                                choices=["gd", "adam", "mom", "rmsprop"],
                                help="name of optimizer")
        g_optimize.add_argument("--momentum", default=None, type=float)
        g_optimize.add_argument("--optimizer_decay", default=None, type=float)
        g_optimize.add_argument("--optimizer_epsilon", default=None, type=float)

        g_rst = parser.add_argument_group(f"({name}) Saver(Restore) Arguments")
        g_rst.add_argument("--trainable_scopes", default="", type=str,
                           help=(
                               "Prefix scopes for training variables (comma separated)\n"
                               "Usually Logits e.g. InceptionResnetV2/Logits/Logits,InceptionResnetV2/AuxLogits/Logits"
                               "For default value, trainable_scopes='' means training 'all' variable"
                               "If you don't want to train(e.g. validation only), "
                               "you should give unmatched random string"
                           ))

        g_options = parser.add_argument_group(f"({name}) Training options(step, batch_size, path) Arguments")

        g_options.add_argument("--no-write_pbtxt", dest="write_pbtxt", action="store_false")
        g_options.add_argument("--write_pbtxt", dest="write_pbtxt", action="store_true",
                               help="write_pbtxt model parameters")
        g_options.set_defaults(write_pbtxt=True)
        g_options.add_argument("--train_dir", required=True, type=str,
                               help="Directory where to write event logs and checkpoint.")
        g_options.add_argument("--step_save_summaries", default=10, type=int)
        g_options.add_argument("--step_save_verbose_summaries", default=2000, type=int)
        g_options.add_argument("--step_save_first_n_summaries", default=30, type=int)
        g_options.add_argument("--step_save_checkpoint", default=500, type=int)
        g_options.add_argument("--step_evaluation", default=500, type=utils.positive_int)

        g_options.add_argument("--max_to_keep", default=5, type=utils.positive_int)
        g_options.add_argument("--max_outputs", default=5, type=utils.positive_int)
        g_options.add_argument("--max_epoch_from_restore", default=50000, type=float,
                               help=(
                                   "max epoch(1 epoch = whole data): "
                                   "Default value for max_epoch is ImageNet Resnet50 max epoch counts"
                               ))
        g_options.add_argument("--step_min_summaries", default=0, type=int)
        g_options.add_argument("--max_step_from_restore", default=sys.maxsize, type=int,
                               help="Stop training when reaching given step value.")
        g_options.add_argument("--tag", default="tag", type=str, help="tag for folder name")
        g_options.add_argument("--no-testmode", dest="testmode", action="store_false")
        g_options.add_argument("--testmode", dest="testmode", action="store_true",
                               help="If testmode, ask deleting train_dir when you exit the process")
        g_options.set_defaults(testmode=False)
        g_options.add_argument("--no-debug", dest="debug", action="store_false")
        g_options.add_argument("--debug", dest="debug", action="store_true", help="Debug model parameters")
        g_options.set_defaults(debug=False)

        g_options.add_argument("--no-step1_mode", dest="step1_mode", action="store_false")
        g_options.add_argument("--step1_mode", dest="step1_mode", action="store_true")
        g_options.set_defaults(step1_mode=False)

        g_options.add_argument("--no-save_evaluation_image", dest="save_evaluation_image", action="store_false")
        g_options.add_argument("--save_evaluation_image", dest="save_evaluation_image", action="store_true")
        g_options.set_defaults(save_evaluation_image=False)

        lr_scheduler.add_arguments(parser)


class MattingTrainer(TrainerBase, MattingBase):
    def __init__(self, model, session, args, dataset, dataset_name):
        super().__init__(model, session, args, dataset, dataset_name, "MattingTrainer")

        self.setup_trainer()
        self.setup_metric_manager()
        self.setup_metric_ops()

        self.routine_experiment_summary()
        self.routine_logging_checkpoint_path()

    def build_evaluate_iterations(self, iters):
        if iters is not None:
            iters = iters
        elif self.args.evaluation_iterations is not None:
            iters = self.args.evaluation_iterations
        else:
            iters = 10  # a default value we used

        return iters

    def setup_metric_manager(self):
        self.metric_manager = metric_manager.MattingMetricManager(
            is_training=True,
            save_evaluation_image=self.args.save_evaluation_image,
            exclude_metric_names=self.args.exclude_metric_names,
            summary=Summaries(
                session=self.session,
                train_dir=self.args.train_dir,
                is_training=True,
                max_image_outputs=self.args.max_image_outputs
            ),
        )

    def setup_metric_ops(self):
        losses = self.build_basic_loss_ops()
        summary_images = self.build_basic_image_ops()
        misc_images = self.build_misc_image_ops()

        self.metric_tf_op = self.metric_manager.build_metric_ops({
            "dataset_split_name": self.dataset_name,
            "target_eval_shape": self.args.target_eval_shape,

            "losses": losses,
            "summary_images": summary_images,
            "misc_images": misc_images,
            "masks": self.model.masks,
            "masks_original": self.model.masks_original,
            "probs": self.model.prob_scores,
        })

    def build_non_tensor_data_from_eval_dict(self, eval_dict, **kwargs):
        return {
            "dataset_split_name": self.dataset_name,

            "batch_infer_time": eval_dict["batch_infer_time"],
            "unit_infer_time": eval_dict["unit_infer_time"],
            "misc_images": None,
            "image_save_dir": None,
        }

    @staticmethod
    def add_arguments(parser):
        g_etc = parser.add_argument_group("(MattingTrainer) ETC")
        # EVALUATE
        g_etc.add_argument("--first_n", default=10, type=utils.positive_int, help="Argument for tf.Print")
