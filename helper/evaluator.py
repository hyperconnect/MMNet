from pathlib import Path
from abc import abstractmethod
import sys

import numpy as np
import tensorflow as tf
from scipy.misc import imsave
from tqdm import tqdm

import common.tf_utils as tf_utils
import metrics.manager as metric_manager
from common.model_loader import Ckpt
from common.utils import format_text
from common.utils import get_logger
from helper.base import Base
from helper.base import MattingBase
from metrics.summaries import BaseSummaries
from metrics.summaries import Summaries


class Evaluator(object):
    _available_inference_output = None

    def __init__(self, model, session, args, dataset, dataset_name, name):
        self.log = get_logger(name)

        self.model = model
        self.session = session
        self.args = args
        self.dataset = dataset
        self.dataset_name = dataset_name

        if Path(self.args.checkpoint_path).is_dir():
            latest_checkpoint = tf.train.latest_checkpoint(self.args.checkpoint_path)
            if latest_checkpoint is not None:
                self.args.checkpoint_path = latest_checkpoint
            self.log.info(f"Get latest checkpoint and update to it: {self.args.checkpoint_path}")

        self.watch_path = self._build_watch_path()

        self.variables_to_restore = Base.get_variables_to_restore(args=self.args, log=self.log, debug=False)
        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.local_variables_initializer())

        self.ckpt_loader = Ckpt(
            session=session,
            variables_to_restore=self.variables_to_restore,
            include_scopes=args.checkpoint_include_scopes,
            exclude_scopes=args.checkpoint_exclude_scopes,
            ignore_missing_vars=args.ignore_missing_vars,
        )

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
    def setup_dataset_iterator(self):
        raise NotImplementedError

    @abstractmethod
    def save_inference_result(self, eval_dict, checkpoint_path):
        raise NotImplementedError

    def _build_watch_path(self):
        if Path(self.args.checkpoint_path).is_dir():
            return Path(self.args.checkpoint_path)
        else:
            return Path(self.args.checkpoint_path).parent

    def build_evaluation_step(self, checkpoint_path):
        if "-" in checkpoint_path and checkpoint_path.split("-")[-1].isdigit():
            return int(checkpoint_path.split("-")[-1])
        else:
            return 0

    def build_checkpoint_paths(self, checkpoint_path):
        checkpoint_glob = Path(checkpoint_path + "*")
        checkpoint_path = Path(checkpoint_path)

        return checkpoint_glob, checkpoint_path

    def build_miscellaneous_path(self, name):
        target_dir = self.watch_path / "miscellaneous" / self.dataset_name / name

        if not target_dir.exists():
            target_dir.mkdir(parents=True)

        return target_dir

    def build_inference_path(self, checkpoint_path):
        if not isinstance(checkpoint_path, Path):
            checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.is_dir():
            root_dir = checkpoint_path
        else:
            root_dir = checkpoint_path.parent

        output_parent_dir = root_dir / "inference" / checkpoint_path.name
        if self.args.inference_output_dirname is not None:
            output_dir = output_parent_dir / self.args.inference_output_dirname
        else:
            output_dir = output_parent_dir / self.args.inference_output

        return output_dir

    def setup_best_keeper(self):
        metric_with_modes = self.metric_manager.get_best_keep_metric_with_modes()
        self.log.debug(metric_with_modes)
        self.best_keeper = tf_utils.BestKeeper(metric_with_modes,
                                               self.dataset_name,
                                               self.watch_path,
                                               self.log)

    def inference(self, checkpoint_path):
        assert not self.args.shuffle, "Current implementation of `inference` requires non-shuffled dataset"
        _data_num = self.dataset.num_samples
        if _data_num % self.args.batch_size != 0:
            with format_text("red", attrs=["bold"]) as fmt:
                self.log.warning(fmt(f"Among {_data_num} data, last {_data_num%self.dataset.batch_size} items will not"
                                     f" be processed during inferential procedure."))

        if self.args.inference_output not in self._available_inference_output:
            raise ValueError(f"Inappropriate inference_output type for "
                             f"{self.__class__.__name__}: {self.args.inference_output}.\n"
                             f"Available outputs are {self._available_inference_output}")

        self.log.info("Inference started")
        self.setup_dataset_iterator()
        self.ckpt_loader.load(checkpoint_path)

        step = self.build_evaluation_step(checkpoint_path)
        checkpoint_glob, checkpoint_path = self.build_checkpoint_paths(checkpoint_path)
        self.session.run(tf.local_variables_initializer())

        eval_dict = self.run_inference(step, is_training=False, do_eval=False)

        self.save_inference_result(eval_dict, checkpoint_path)

    def evaluate_once(self, checkpoint_path):
        self.log.info("Evaluation started")
        self.setup_dataset_iterator()
        self.ckpt_loader.load(checkpoint_path)

        step = self.build_evaluation_step(checkpoint_path)
        checkpoint_glob, checkpoint_path = self.build_checkpoint_paths(checkpoint_path)
        self.session.run(tf.local_variables_initializer())

        eval_metric_dict = self.run_evaluation(step, is_training=False)
        best_keep_metric_dict = self.metric_manager.filter_best_keep_metric(eval_metric_dict)
        is_keep, metrics_keep = self.best_keeper.monitor(self.dataset_name, best_keep_metric_dict)

        if self.args.save_best_keeper:
            meta_info = {
                "step": step,
                "model_size": self.model.total_params,
            }
            self.best_keeper.remove_old_best(self.dataset_name, metrics_keep)
            self.best_keeper.save_best(self.dataset_name, metrics_keep, checkpoint_glob)
            self.best_keeper.remove_temp_dir()
            self.best_keeper.save_scores(self.dataset_name, metrics_keep, best_keep_metric_dict, meta_info)

        self.metric_manager.write_evaluation_summaries(step=step,
                                                       collection_keys=[BaseSummaries.KEY_TYPES.DEFAULT])
        self.metric_manager.log_metrics(step=step)

        self.log.info("Evaluation finished")

        if step >= self.args.max_step_from_restore:
            self.log.info("Evaluation stopped")
            sys.exit()

    def build_train_directory(self):
        if Path(self.args.checkpoint_path).is_dir():
            return str(self.args.checkpoint_path)
        else:
            return str(Path(self.args.checkpoint_path).parent)

    @staticmethod
    def add_arguments(parser):
        g = parser.add_argument_group("(Evaluator) arguments")

        g.add_argument(
            "--inference_output",
            type=str,
            default="none",
        )
        g.add_argument(
            "--inference_output_dirname",
            type=str,
            default=None,
        )

        g.add_argument("--valid_type", default="loop", type=str, choices=["loop", "once"])
        g.add_argument("--max_outputs", default=5, type=int)

        g.add_argument("--no-convert_to_pb", dest="convert_to_pb", action="store_false")
        g.add_argument("--convert_to_pb", dest="convert_to_pb", action="store_true")
        g.set_defaults(convert_to_pb=True)

        g.add_argument("--no-save_evaluation_image", dest="save_evaluation_image", action="store_false")
        g.add_argument("--save_evaluation_image", dest="save_evaluation_image", action="store_true")
        g.set_defaults(save_evaluation_image=False)

        g.add_argument("--no-save_best_keeper", dest="save_best_keeper", action="store_false")
        g.add_argument("--save_best_keeper", dest="save_best_keeper", action="store_true")
        g.set_defaults(save_best_keeper=True)

        g.add_argument("--max_step_from_restore", default=1e20, type=int)


class MattingEvaluator(Evaluator, MattingBase):
    _available_inference_output = ["image_under_prob", "binary_mask", "prob_mask", "image_and_mask"]

    def __init__(self, model, session, args, dataset, dataset_name):
        super().__init__(model, session, args, dataset, dataset_name, "MattingEvaluator")

        self.setup_metric_manager()
        self.setup_metric_ops()
        self.setup_best_keeper()

    def setup_metric_manager(self):
        self.metric_manager = metric_manager.MattingMetricManager(
            is_training=False,
            save_evaluation_image=self.args.save_evaluation_image,
            exclude_metric_names=self.args.exclude_metric_names,
            summary=Summaries(
                session=self.session,
                train_dir=self.build_train_directory(),
                is_training=False,
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

    def setup_dataset_iterator(self):
        self.dataset.setup_iterator(
            self.session,
            (self.dataset.image_placeholder, self.dataset.mask_placeholder),
            (self.dataset.image_fullpathes, self.dataset.mask_fullpathes),
        )

    def build_non_tensor_data_from_eval_dict(self, eval_dict, **kwargs):
        return {
            "dataset_split_name": self.dataset_name,

            "batch_infer_time": eval_dict["batch_infer_time"],
            "unit_infer_time": eval_dict["unit_infer_time"],
            "misc_images": dict(filter(lambda x: x[0].startswith("misc_images/"), eval_dict.items())),
            "image_save_dir": self.build_miscellaneous_path("images"),
        }

    def save_inference_result(self, eval_dict, checkpoint_path):
        output_dir = self.build_inference_path(checkpoint_path)
        if not output_dir.is_dir():
            output_dir.mkdir(parents=True)
            self.log.info(f"Make directory {output_dir}")

        self.log.info(f"Inference results will be saved under {output_dir}")

        predictions = eval_dict["probs"]
        images = eval_dict["images"]

        for prediction, image, image_fullpath in tqdm(zip(predictions, images, self.dataset.image_fullpathes)):
            prediction = prediction[:, :, 1:]
            imagename = Path(image_fullpath).name
            prediction_normed = (prediction - prediction.min()) / (prediction.max() - prediction.min())
            if self.args.inference_output == "image_under_prob":
                _output = np.squeeze(np.expand_dims(prediction_normed, 0) * image)
            elif self.args.inference_output == "binary_mask":
                _output = np.squeeze(prediction_normed > 0.5) * 255
            elif self.args.inference_output == "prob_mask":
                _output = np.squeeze(prediction_normed) * 255
            elif self.args.inference_output == "image_and_mask":
                _output = np.concatenate([image, image * prediction, np.tile(prediction * 255, 3)], axis=0)
            _output = _output.round().astype(np.uint8)
            imsave(output_dir / imagename, _output)

        self.log.info(f"Inference results saved under {output_dir}")
