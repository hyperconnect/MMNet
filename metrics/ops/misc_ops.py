import numpy as np
from PIL import Image

from metrics.ops.base_ops import NonTensorMetricOpBase
from metrics.ops.base_ops import TensorMetricOpBase
from metrics.summaries import BaseSummaries
import metrics.parser as parser


class InferenceTimeMetricOp(NonTensorMetricOpBase):
    """
    Inference Time Metric.
    """
    _properties = {
        "is_for_summary": True,
        "is_for_best_keep": False,
        "is_for_log": True,
        "valid_input_data_parsers": [
            parser.MattingDataParser,
        ],
        "summary_collection_key": BaseSummaries.KEY_TYPES.DEFAULT,
        "summary_value_type": BaseSummaries.VALUE_TYPES.PLACEHOLDER,
        "min_max_mode": None,
    }

    def __str__(self):
        return "inference_time_metric"

    def build_op(self,
                 data):
        return {
            f"misc/batch_infer_time/{data.dataset_split_name}": None,
            f"misc/unit_infer_time/{data.dataset_split_name}": None,
        }

    def evaluate(self,
                 data):
        return {
            f"misc/batch_infer_time/{data.dataset_split_name}": np.mean(data.batch_infer_time),
            f"misc/unit_infer_time/{data.dataset_split_name}": np.mean(data.unit_infer_time),
        }


class MiscImageRetrieveOp(TensorMetricOpBase):
    """ Image not recoreded on summary but to be retrieved.
    """
    _properties = {
        "is_for_summary": False,
        "is_for_best_keep": False,
        "is_for_log": False,
        "valid_input_data_parsers": [
            parser.MattingDataParser,
        ],
        "summary_collection_key": None,
        "summary_value_type": None,
        "min_max_mode": None,
    }

    def __str__(self):
        return "misc_image_retrieve"

    def build_op(self, data):
        result = dict()

        for name, op in data.misc_images.items():
            key = f"misc_images/{data.dataset_split_name}/{name}"
            result[key] = op

        return result

    def expectation_of(self, data):
        # we don't aggregate this output
        pass


class MiscImageSaveOp(NonTensorMetricOpBase):
    """ Image save
    """
    _properties = {
        "is_for_summary": False,
        "is_for_best_keep": False,
        "is_for_log": True,
        "valid_input_data_parsers": [
            parser.MattingDataParser,
        ],
        "summary_collection_key": None,
        "summary_value_type": None,
        "min_max_mode": None,
    }

    def __str__(self):
        return "misc_image_save"

    def build_op(self, data):
        return {
            f"save_images/{data.dataset_split_name}": None
        }

    def evaluate(self, data):
        keys = []
        images = []
        for key, image in data.misc_images.items():
            keys.append(key)
            images.append(image)

        # meta info
        num_data = images[0].shape[0]
        h, w = images[0].shape[1:3]

        for nidx in range(num_data):
            _images = [Image.fromarray(img[nidx]) for img in images]

            merged_image = Image.new("RGB", (h * ((len(_images)-1) // 3 + 1), w * 3))
            for i, img in enumerate(_images):
                row = i // 3
                col = i % 3

                merged_image.paste(img, (row*w, col*h, (row+1)*w, (col+1)*h))
            merged_image.save(data.image_save_dir / f"img_{nidx}.jpg")

        # msg
        msg = f"{num_data} images are saved under {data.image_save_dir.resolve()}"

        return {
            f"save_images/{data.dataset_split_name}": msg
        }
