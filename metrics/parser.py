from abc import ABC, ABCMeta

import tensorflow as tf

from metrics.base import DataStructure


class MetricDataParserBase(ABC):
    @classmethod
    def parse_build_data(cls, data):
        """
        Args:
            data: dictionary which will be passed to InputBuildData
        """
        data = cls._validate_build_data(data)
        data = cls._process_build_data(data)
        return data

    @classmethod
    def parse_non_tensor_data(cls, data):
        """
        Args:
            data: dictionary which will be passed to InputDataStructure
        """
        input_data = cls._validate_non_tensor_data(data)
        output_data = cls._process_non_tensor_data(input_data)
        return output_data

    @classmethod
    def _validate_build_data(cls, data):
        """
        Specify assertions that tensor data should contains

        Args:
            data: dictionary
        Return:
            InputDataStructure
        """
        return cls.InputBuildData(data)

    @classmethod
    def _validate_non_tensor_data(cls, data):
        """
        Specify assertions that non-tensor data should contains

        Args:
            data: dictionary
        Return:
            InputDataStructure
        """
        return cls.InputNonTensorData(data)

    """
    Override these two functions if needed.
    """
    @classmethod
    def _process_build_data(cls, data):
        """
        Process data in order to following metrics can use it

        Args:
            data: InputBuildData

        Return:
            OutputBuildData
        """
        # default function is just passing data
        return cls.OutputBuildData(data.to_dict())

    @classmethod
    def _process_non_tensor_data(cls, data):
        """
        Process data in order to following metrics can use it

        Args:
            data: InputNonTensorData

        Return:
            OutputNonTensorData
        """
        # default function is just passing data
        return cls.OutputNonTensorData(data.to_dict())

    """
    Belows should be implemented when inherit.
    """
    class InputBuildData(DataStructure, metaclass=ABCMeta):
        pass

    class OutputBuildData(DataStructure, metaclass=ABCMeta):
        pass

    class InputNonTensorData(DataStructure, metaclass=ABCMeta):
        pass

    class OutputNonTensorData(DataStructure, metaclass=ABCMeta):
        pass


class MattingDataParser(MetricDataParserBase):
    """ Matting parser
    """
    class InputBuildData(DataStructure):
        _keys = [
            "dataset_split_name",
            "target_eval_shape",  # Tuple(int, int) | (height, width)

            "losses",  # Dict | loss_key -> Tensor
            "summary_images",  # Dict | summary_name -> Tensor
            "misc_images",  # Dict | name -> Tensor
            "masks",  # Tensor
            "masks_original",  # Tensor
            "probs",  # Tensor
        ]

    class OutputBuildData(DataStructure):
        _keys = [
            "dataset_split_name",
            "target_eval_shape",

            "losses",
            "summary_images",
            "misc_images",
            "masks",
            "alpha_scores",  # Tensor
            "binary_masks",
            "binary_alpha_scores",

            "ts_masks",
            "ts_alpha_scores",  # Tensor
            "ts_binary_masks",
            "ts_binary_alpha_scores",
        ]

    class InputNonTensorData(DataStructure):
        _keys = [
            "dataset_split_name",

            "batch_infer_time",
            "unit_infer_time",
            "misc_images",
            "image_save_dir",
        ]

    class OutputNonTensorData(DataStructure):
        _keys = [
            "dataset_split_name",

            "batch_infer_time",
            "unit_infer_time",
            "misc_images",
            "image_save_dir",
        ]

    @classmethod
    def _process_build_data(cls, data):
        masks = tf.expand_dims(data.masks, axis=3)
        alpha_scores = data.probs[:, :, :, -1:]
        binary_masks = tf.greater_equal(masks, 0.5)
        binary_alpha_scores = tf.greater_equal(alpha_scores, 0.5)

        if data.target_eval_shape:
            ts_masks = tf.image.resize_bilinear(data.masks_original, data.target_eval_shape)
            ts_alpha_scores = tf.image.resize_bilinear(alpha_scores, data.target_eval_shape)

            ts_binary_masks = tf.greater_equal(ts_masks, 0.5)
            ts_binary_alpha_scores = tf.greater_equal(ts_alpha_scores, 0.5)
        else:
            ts_masks = None
            ts_alpha_scores = None

            ts_binary_masks = None
            ts_binary_alpha_scores = None

        return cls.OutputBuildData({
            "dataset_split_name": data.dataset_split_name,
            "target_eval_shape": data.target_eval_shape,
            "losses": data.losses,
            "summary_images": data.summary_images,
            "misc_images": data.misc_images,

            "masks": masks,
            "alpha_scores": alpha_scores,
            "binary_masks": binary_masks,
            "binary_alpha_scores": binary_alpha_scores,

            "ts_masks": ts_masks,
            "ts_alpha_scores": ts_alpha_scores,
            "ts_binary_masks": ts_binary_masks,
            "ts_binary_alpha_scores": ts_binary_alpha_scores,
        })

    @classmethod
    def _process_non_tensor_data(cls, data):
        return cls.OutputNonTensorData({
            "dataset_split_name": data.dataset_split_name,
            "batch_infer_time": data.batch_infer_time,
            "unit_infer_time": data.unit_infer_time,
            "misc_images": data.misc_images,
            "image_save_dir": data.image_save_dir,
        })
