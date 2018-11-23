import tensorflow as tf
import tensorflow.contrib.slim as slim

import common.utils as utils
import factory.losses_funcs as losses_funcs
import matting_nets.deeplab_v3plus as deeplab
from matting_nets import mmnet
from factory.base import CNNModel
from factory.matting_converter import ProbConverter


_available_nets = [
    "MMNetModel",
    "DeepLabModel",
]


class MattingNetModel(CNNModel):
    def __init__(self, args, dataset=None):
        self.log = utils.get_logger("MattingNetModel")
        self.args = args
        self.dataset = dataset  # used to access data created in DataWrapper

    def build(
        self,
        images_original: tf.Tensor,
        images: tf.Tensor,
        masks_original: tf.Tensor,
        masks: tf.Tensor,
        is_training: bool,
    ):
        self._images_original = images_original
        self._images = images
        self._masks = masks
        self._masks_original = masks_original

        # -- * -- build_output: it includes build_inference
        inputs, logits, outputs, self.endpoints = self.build_output(
            self.images,
            is_training,
            self.args.output_name,
        )
        self.prob_scores = outputs

        # -- * -- Build loss function: it can be different between each models
        self._total_loss, self._model_loss, self.endpoints_loss = self.build_loss(
            inputs,
            logits,
            outputs,
            self.masks,
        )

        # -- * -- build iamge ops for summary
        self._image_ops = self.build_image_ops(self.images, self.prob_scores, self.masks)

        self.total_params = self.build_finish(is_training, self.log)

    @property
    def total_loss(self):
        return self._total_loss

    @property
    def model_loss(self):
        return self._model_loss

    def build_output(
        self,
        images,
        is_training,
        output_name,
    ):
        inputs = self.build_images(images)
        logits, endpoints = self.build_inference(
            inputs,
            is_training=is_training,
        )
        outputs = ProbConverter.convert(
            logits,
            output_name,
            self.args.num_classes,
        )
        return inputs, logits, outputs, endpoints

    def build_images(
        self,
        images: tf.Tensor,
    ):
        images_preprocessed = self.preprocess_images(
            images,
            preprocess_method=self.args.preprocess_method,
        )

        return images_preprocessed

    def build_inference(self, images, is_training=True):
        raise NotImplementedError

    def build_loss(
        self,
        inputs: tf.Tensor,
        logits: tf.Tensor,
        scores: tf.Tensor,
        masks: tf.Tensor,
    ):
        endpoints_loss = {}
        alpha_scores = scores[:, :, :, -1:]
        masks_float32 = tf.expand_dims(tf.cast(masks, dtype=tf.float32), axis=3)

        sum_lambda = 0.

        # alpha loss
        if self.args.lambda_alpha_loss > 0:
            endpoints_loss["alpha_loss"] = losses_funcs.alpha_loss(
                alpha_scores=alpha_scores,
                masks_float32=masks_float32,
            )
            endpoints_loss["alpha_loss"] *= self.args.lambda_alpha_loss
            sum_lambda += self.args.lambda_alpha_loss

        # compositional loss
        if self.args.lambda_comp_loss > 0:
            endpoints_loss["comp_loss"] = losses_funcs.comp_loss(
                alpha_scores=alpha_scores,
                masks_float32=masks_float32,
                images=inputs,
            )
            endpoints_loss["comp_loss"] *= self.args.lambda_comp_loss
            sum_lambda += self.args.lambda_comp_loss

        # gradient loss
        if self.args.lambda_grad_loss > 0:
            endpoints_loss["grad_loss"] = losses_funcs.grad_loss(
                alpha_scores=alpha_scores,
                masks_float32=masks_float32,
            )
            endpoints_loss["grad_loss"] *= self.args.lambda_grad_loss
            sum_lambda += self.args.lambda_grad_loss

        # kd loss
        if self.args.lambda_kd_loss > 0:
            endpoints_loss["kd_loss"] = losses_funcs.kd_loss(
                scores=logits,
                masks=masks,
            )
            endpoints_loss["kd_loss"] *= self.args.lambda_kd_loss
            sum_lambda += self.args.lambda_kd_loss

        # kd aux loss
        if self.args.lambda_aux_loss > 0:
            _, h, w, _ = self.encoded_map.get_shape().as_list()
            compress_mask = tf.image.resize_images(
                tf.expand_dims(masks, 3),
                [h, w],
                tf.image.ResizeMethod.NEAREST_NEIGHBOR
            )
            compress_mask = tf.squeeze(compress_mask, 3)

            endpoints_loss["aux_loss"] = losses_funcs.kd_loss(
                scores=self.encoded_map,
                masks=compress_mask,
            )
            endpoints_loss["aux_loss"] *= self.args.lambda_aux_loss
            sum_lambda += self.args.lambda_aux_loss

        model_loss = tf.add_n(list(endpoints_loss.values())) / sum_lambda

        if len(tf.losses.get_regularization_losses()) > 0:
            reg_loss = tf.add_n(tf.losses.get_regularization_losses())
        else:
            reg_loss = tf.constant(0.)

        endpoints_loss.update({
            "regularize_loss": reg_loss,
        })

        total_loss = model_loss + reg_loss
        return total_loss, model_loss, endpoints_loss

    @staticmethod
    def build_image_ops(images, prob_scores, masks):
        alpha_scores = prob_scores[:, :, :, -1:]
        binary_scores = tf.cast((alpha_scores > 0.5), tf.float32)
        masks_casted = tf.expand_dims(masks, -1)
        images_under_binary = images * binary_scores
        images_under_prob = images * alpha_scores

        image_ops = {
            "images": images,
            "prob_scores": alpha_scores * 255,
            "binary_scores": binary_scores * 255,
            "images_under_binary": images_under_binary,
            "images_under_prob": images_under_prob,
            "masks": masks_casted * 255,
        }

        return {k: tf.cast(op, tf.uint8) for k, op in image_ops.items()}

    @property
    def images_original(self):
        return self._images_original

    @property
    def images(self):
        return self._images

    @property
    def image_ops(self):
        return self._image_ops

    @property
    def masks_original(self):
        return self._masks_original

    @property
    def masks(self):
        return self._masks

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--lambda_alpha_loss", type=float, default=0.0)
        parser.add_argument("--lambda_comp_loss", type=float, default=0.0)
        parser.add_argument("--lambda_grad_loss", type=float, default=0.0)
        parser.add_argument("--lambda_kd_loss", type=float, default=0.0)
        parser.add_argument("--lambda_aux_loss", type=float, default=0.0)


class MMNetModel(MattingNetModel):
    def __init__(self, args, dataset=None):
        super().__init__(args, dataset)

    def build_inference(self, images, is_training=True):
        with slim.arg_scope(mmnet.MMNet_arg_scope(use_fused_batchnorm=self.args.use_fused_batchnorm,
                                                  weight_decay=self.args.weight_decay,
                                                  dropout=self.args.dropout)):
            logit_scores, endpoints = mmnet.MMNet(
                images,
                is_training,
                depth_multiplier=self.args.width_multiplier,
            )
        self.encoded_map = endpoints["aux_block"]
        return logit_scores, endpoints

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--width_multiplier", default=1.0, type=float)
        parser.add_argument("--weight_decay", type=float, default=0.0)
        parser.add_argument("--dropout", type=float, default=0.0)


class DeepLabModel(MattingNetModel):
    def __init__(self, args, dataset=None):
        super(DeepLabModel, self).__init__(args, dataset)

    def build_inference(self, images, is_training=True):
        with slim.arg_scope(deeplab.DeepLab_arg_scope(weight_decay=self.args.weight_decay)):
            logit_scores, endpoints = deeplab.DeepLab(images,
                                                      self.args,
                                                      is_training=is_training,
                                                      depth_multiplier=self.args.depth_multiplier,
                                                      output_stride=self.args.output_stride)
        return logit_scores, endpoints

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--extractor", type=str, required=True, choices=deeplab.available_extractors)
        parser.add_argument("--weight_decay", type=float, default=0.00004)
        parser.add_argument("--depth_multiplier", default=1.0, type=float, help="MobileNetV2 depth_multiplier")
        parser.add_argument("--output_stride", default=16, type=int)
