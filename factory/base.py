from abc import ABC
from abc import abstractmethod

import tensorflow as tf
import tensorflow.contrib.slim as slim

import common.tf_utils as tf_utils


class CNNModel(ABC):
    def preprocess_images(self, images, preprocess_method, reuse=False):
        with tf.variable_scope("preprocess", reuse=reuse):
            if images.dtype == tf.uint8:
                images = tf.cast(images, tf.float32)
            if preprocess_method == "preprocess_normalize":
                # -- * -- preprocess_normalize
                # Scale input images to range [0, 1], same scales like mean of masks
                images = tf.divide(images, tf.constant(255.0))
            elif preprocess_method == "no_preprocessing":
                pass
            else:
                raise ValueError("Unsupported preprocess_method: {}".format(preprocess_method))

        return images

    @staticmethod
    def add_arguments(parser, default_type):
        g_cnn = parser.add_argument_group("(CNNModel) Arguments")
        assert default_type in ["matting", None]

        g_cnn.add_argument("--task_type", type=str, required=True,
                           choices=[
                               "matting",
                           ])
        g_cnn.add_argument("--num_classes", type=int, default=None,
                           help=(
                               "It is currently not used in multi-task learning, "
                               "so it can't *required*"
                           ))
        g_cnn.add_argument("--checkpoint_path", default="", type=str)

        g_cnn.add_argument("--input_name", type=str, default="input/image")
        g_cnn.add_argument("--input_batch_size", type=int, default=1)
        g_cnn.add_argument("--output_name", type=str, required=True)
        g_cnn.add_argument("--output_type", type=str, help="mainly used in convert.py", required=True)

        g_cnn.add_argument("--no-use_fused_batchnorm", dest="use_fused_batchnorm", action="store_false")
        g_cnn.add_argument("--use_fused_batchnorm", dest="use_fused_batchnorm", action="store_true")
        g_cnn.set_defaults(use_fused_batchnorm=True)

        g_cnn.add_argument("--verbosity", default=0, type=int,
                           help="If verbosity > 0, then summary batch_norm scalar metrics etc")
        g_cnn.add_argument("--preprocess_method", required=True, type=str,
                           choices=["no_preprocessing", "preprocess_normalize"])

        g_cnn.add_argument("--no-ignore_missing_vars", dest="ignore_missing_vars", action="store_false")
        g_cnn.add_argument("--ignore_missing_vars", dest="ignore_missing_vars", action="store_true")
        g_cnn.set_defaults(ignore_missing_vars=False)

        g_cnn.add_argument("--checkpoint_exclude_scopes", default="", type=str,
                           help=("Prefix scopes that shoule be EXLUDED for restoring variables "
                                 "(comma separated)\n Usually Logits e.g. InceptionResnetV2/Logits/Logits, "
                                 "InceptionResnetV2/AuxLogits/Logits"))

        g_cnn.add_argument("--checkpoint_include_scopes", default="", type=str,
                           help=("Prefix scopes that should be INCLUDED for restoring variables "
                                 "(comma separated)"))

    def build_finish(self, is_training, log):
        total_params = tf_utils.show_models(log)

        if self.args.verbosity >= 1:
            slim.model_analyzer.analyze_ops(tf.get_default_graph(), print_info=True)

        return total_params

    @abstractmethod
    def build_output(self):
        pass

    @property
    @abstractmethod
    def images(self):
        pass

    @property
    @abstractmethod
    def images_original(self):
        pass

    @property
    @abstractmethod
    def total_loss(self):
        pass

    @property
    @abstractmethod
    def model_loss(self):
        pass
