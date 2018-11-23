import tensorflow as tf
import numpy as np
from overload import overload

import common.image_helper as image_helper
import metrics.parser as parser
from metrics.ops.base_ops import TensorMetricOpBase
from metrics.summaries import BaseSummaries


class LossesMetricOp(TensorMetricOpBase):
    """ Loss Metric.
    """
    _properties = {
        "is_for_summary": True,
        "is_for_best_keep": True,
        "is_for_log": True,
        "valid_input_data_parsers": [
            parser.MattingDataParser,
        ],
        "summary_collection_key": BaseSummaries.KEY_TYPES.DEFAULT,
        "summary_value_type": BaseSummaries.VALUE_TYPES.PLACEHOLDER,
        "min_max_mode": "min",
    }

    def __str__(self):
        return "losses"

    def build_op(self, data):
        result = dict()

        for loss_name, loss_op in data.losses.items():
            key = f"metric_loss/{data.dataset_split_name}/{loss_name}"
            result[key] = loss_op

        return result

    def expectation_of(self, data: np.array):
        assert len(data.shape) == 2
        return np.mean(data)


class ImageSummaryOp(TensorMetricOpBase):
    """ Image summary
    """
    _properties = {
        "is_for_summary": True,
        "is_for_best_keep": False,
        "is_for_log": False,
        "valid_input_data_parsers": [
            parser.MattingDataParser,
        ],
        "summary_collection_key": BaseSummaries.KEY_TYPES.DEFAULT,
        "summary_value_type": BaseSummaries.VALUE_TYPES.IMAGE,
        "min_max_mode": None,
    }

    def __str__(self):
        return "summary_images"

    @overload
    def build_op(self,
                 data: parser.MattingDataParser.OutputBuildData):
        result = dict()

        for summary_name, op in data.summary_images.items():
            key = f"{summary_name}/{data.dataset_split_name}"
            result[key] = op

        return result

    def expectation_of(self, data):
        pass


class MADMetricOp(TensorMetricOpBase):
    """ Mean Average Difference Metric.
    """
    _properties = {
        "is_for_summary": True,
        "is_for_best_keep": True,
        "is_for_log": True,
        "valid_input_data_parsers": [
            parser.MattingDataParser,
        ],
        "summary_collection_key": BaseSummaries.KEY_TYPES.DEFAULT,
        "summary_value_type": BaseSummaries.VALUE_TYPES.PLACEHOLDER,
        "min_max_mode": "min",
    }

    def __str__(self):
        return "mad_metric"

    @overload
    def build_op(self,
                 data: parser.MattingDataParser.OutputBuildData):
        def _calc(masks, alpha_scores):
            return tf.reduce_mean(tf.abs(masks - alpha_scores))

        result = dict()

        result[f"MAD/{data.dataset_split_name}"] = _calc(data.masks, data.alpha_scores)

        if data.target_eval_shape:
            suffix = f"{data.target_eval_shape[0]}_{data.target_eval_shape[1]}"
            result[f"MAD/{data.dataset_split_name}/{suffix}"] = _calc(data.ts_masks, data.ts_alpha_scores)

        return result

    def expectation_of(self, data: np.array):
        assert len(data.shape) == 2
        return np.mean(data)


class GaussianGradMetricOp(TensorMetricOpBase):
    """ Gradient Error Metric.
    """
    _properties = {
        "is_for_summary": True,
        "is_for_best_keep": True,
        "is_for_log": True,
        "valid_input_data_parsers": [
            parser.MattingDataParser,
        ],
        "summary_collection_key": BaseSummaries.KEY_TYPES.DEFAULT,
        "summary_value_type": BaseSummaries.VALUE_TYPES.PLACEHOLDER,
        "min_max_mode": "min",
    }

    def __str__(self):
        return "gaussian_grad_metric"

    @overload
    def build_op(self,
                 data: parser.MattingDataParser.OutputBuildData):
        def _norm(x):
            return tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1))

        def _calc(masks, alpha_scores):
            grad_masks = image_helper.first_deriviate_gaussian_gradient(
                masks,
                sigma=1.4,
            )
            grad_alpha_scores = image_helper.first_deriviate_gaussian_gradient(
                alpha_scores,
                sigma=1.4,
            )

            metric = tf.reduce_mean(_norm(grad_masks - grad_alpha_scores))

            return metric

        result = dict()
        result[f"GAUSS_GRAD/{data.dataset_split_name}"] = _calc(data.masks, data.alpha_scores)

        if data.target_eval_shape:
            suffix = f"{data.target_eval_shape[0]}_{data.target_eval_shape[1]}"
            result[f"GAUSS_GRAD/{data.dataset_split_name}/{suffix}"] = _calc(data.ts_masks, data.ts_alpha_scores)

        return result

    def expectation_of(self, data: np.array):
        assert len(data.shape) == 2
        return np.mean(data)
