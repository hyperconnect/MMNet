from abc import ABC
from abc import abstractmethod
import tensorflow as tf


class ConverterBase(ABC):
    @classmethod
    @abstractmethod
    def convert(
        cls,
        logits: tf.Tensor,
        output_name: str,
        num_classes: int,
    ):
        raise NotImplementedError(f"convert() not defined in {cls.__class__.__name__}")


class ProbConverter(ConverterBase):
    @classmethod
    def convert(
        cls,
        logits: tf.Tensor,
        output_name: str,
        num_classes: int,
    ):
        assert num_classes == 2

        softmax_scores = tf.contrib.layers.softmax(logits, scope="output/softmax")
        # tf.identity to assign output_name
        output = tf.identity(softmax_scores, name=output_name)
        return output

