from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Tuple
from typing import List
from itertools import chain

import tensorflow as tf
from termcolor import colored

import common.utils as utils
from datasets.augmentation_factory import _available_augmentation_methods
from datasets.augmentation_factory import get_augmentation_fn


class DataWrapperBase(ABC):
    def __init__(
        self,
        args,
        dataset_split_name: str,
        is_training: bool,
        name: str,
    ):
        self.name = name
        self.args = args
        self.dataset_split_name = dataset_split_name
        self.is_training = is_training

        # args.inference is False by default
        self.need_label = not self.args.inference
        self.shuffle = self.args.shuffle
        self.supported_extensions = [".jpg", ".JPEG", ".png"]

        self.log = utils.get_logger(self.name, None)
        self.timer = utils.Timer(self.log)
        self.dataset_path = Path(self.args.dataset_path)
        self.dataset_path_with_split_name = self.dataset_path / self.dataset_split_name

        with utils.format_text("yellow", ["underline"]) as fmt:
            self.log.info(self.name)
            self.log.info(fmt(f"dataset_path_with_split_name: {self.dataset_path_with_split_name}"))
            self.log.info(fmt(f"dataset_split_name: {self.dataset_split_name}"))

    @property
    @abstractmethod
    def num_samples(self):
        pass

    @property
    def padded_original_image_dummy_shape(self):
        return (1, 1, 1)

    @property
    def padded_max_size(self):
        return 400

    def resize_and_padding_before_augmentation(self, image, size):
        # If width > height, resize height to model's input height while preserving aspect ratio
        # If height > width, resize width to model's input width while preserving aspect ratio
        if self.args.debug_augmentation:
            assert size[0] == size[1], "resize_and_padding_before_augmentation only supports square target image"
            image = tf.expand_dims(image, 0)

            image_dims = tf.shape(image)
            height = image_dims[1]
            width = image_dims[2]

            min_size = min(*size)
            width_aspect = tf.maximum(min_size, tf.cast(width * min_size / height, dtype=tf.int32))
            height_aspect = tf.maximum(min_size, tf.cast(height * min_size / width, dtype=tf.int32))

            image = tf.image.resize_bilinear(image, (height_aspect, width_aspect))
            image = image[:, :self.padded_max_size, :self.padded_max_size, :]

            # Pads the image on the bottom and right with zeros until it has dimensions target_height, target_width.
            image = tf.image.pad_to_bounding_box(
                image,
                offset_height=tf.maximum(self.padded_max_size-height_aspect, 0),
                offset_width=tf.maximum(self.padded_max_size-width_aspect, 0),
                target_height=self.padded_max_size,
                target_width=self.padded_max_size,
            )

            image = tf.squeeze(image, 0)
            return image
        else:
            # Have to return some dummy tensor which have .get_shape() to tf.dataset
            return tf.constant(0, shape=self.padded_original_image_dummy_shape, dtype=tf.uint8, name="dummy")

    def augment_image(self, image, args, channels=3, **kwargs):
        aug_fn = get_augmentation_fn(args.augmentation_method)
        image = aug_fn(image, args.height, args.width, channels, **kwargs)
        return image

    @property
    def batch_size(self):
        try:
            return self._batch_size
        except AttributeError:
            self._batch_size = 0

    @batch_size.setter
    def batch_size(self, val):
        self._batch_size = val

    def get_all_images(self, image_path):
        if isinstance(image_path, list):
            img_gen = []
            for p in image_path:
                for ext in self.supported_extensions:
                    img_gen.append(Path(p).glob(f"*{ext}"))
        else:
            img_gen = [Path(image_path).glob(f"*{ext}") for ext in self.supported_extensions]

        return chain(*img_gen)

    def setup_dataset(
        self,
        placeholders: Tuple[tf.placeholder, tf.placeholder],
        batch_size: int=None,
    ):
        self.batch_size = self.args.batch_size if batch_size is None else batch_size

        dataset = tf.data.Dataset.from_tensor_slices(placeholders)
        dataset = dataset.map(self._parse_function, num_parallel_calls=self.args.num_threads).prefetch(
            self.args.prefetch_factor * self.batch_size)
        if self.is_training:
            dataset = dataset.repeat()
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=self.args.buffer_size)
        self.dataset = dataset.batch(self.batch_size)
        self.iterator = self.dataset.make_initializable_iterator()
        self.next_elem = self.iterator.get_next()

    def setup_iterator(self,
                       session: tf.Session,
                       placeholders: Tuple[tf.placeholder, tf.placeholder],
                       variables: Tuple[tf.placeholder, tf.placeholder],
                       ):
        assert len(placeholders) == len(variables), "Length of placeholders and variables differ!"
        with self.timer(colored("Initialize data iterator.", "yellow")):
            session.run(self.iterator.initializer,
                        feed_dict={placeholder: variable for placeholder, variable in zip(placeholders, variables)})

    def get_input_and_output_op(self):
        return self.next_elem

    def __str__(self):
        return f"path: {self.args.dataset_path}, split: {self.args.dataset_split_name} data size: {self._num_samples}"

    def get_all_dataset_paths(self) -> List[str]:
        if self.args.has_sub_dataset:
            return sorted([p for p in self.dataset_path_with_split_name.glob("*/") if p.is_dir()])
        else:
            return [self.dataset_path_with_split_name]

    @staticmethod
    def add_arguments(parser):
        g_common = parser.add_argument_group("(DataWrapperBase) Common Arguments for all data wrapper.")
        g_common.add_argument("--dataset_path", required=True, type=str, help="The name of the dataset to load.")
        g_common.add_argument("--dataset_split_name", required=True, type=str, nargs="*",
                              help="The name of the train/test split. Support multiple splits")

        g_common.add_argument("--batch_size", default=32, type=utils.positive_int,
                              help="The number of examples in batch.")
        g_common.add_argument("--no-shuffle", dest="shuffle", action="store_false")
        g_common.add_argument("--shuffle", dest="shuffle", action="store_true")
        g_common.set_defaults(shuffle=True)

        g_common.add_argument("--width", required=True, type=int)
        g_common.add_argument("--height", required=True, type=int)
        g_common.add_argument("--no-debug_augmentation", dest="debug_augmentation", action="store_false")
        g_common.add_argument("--debug_augmentation", dest="debug_augmentation", action="store_true")
        g_common.set_defaults(debug_augmentation=False)
        g_common.add_argument("--max_padded_size", default=224, type=int,
                              help=("We will resize & pads the original image "
                                    "until it has dimensions (padded_size, padded_size)"
                                    "Recommend to set this value as width(or height) * 1.8 ~ 2"))
        g_common.add_argument("--augmentation_method", type=str, required=True,
                              choices=_available_augmentation_methods)
        g_common.add_argument("--num_threads", default=8, type=int)
        g_common.add_argument("--buffer_size", default=1000, type=int)
        g_common.add_argument("--prefetch_factor", default=100, type=int)

        g_common.add_argument("--rotation_range", default=0, type=int,
                              help="Receives maximum angle to be rotated in terms of degree: "
                                   "The image is randomly rotated by the angle "
                                   "randomly chosen from [-rotation_range, rotation_range], "
                                   "and then cropped appropriately to remove dark areas.\n"
                                   "So, be aware that the rotation performs certain kind of zooming.")
        g_common.add_argument("--no-has_sub_dataset", dest="has_sub_dataset", action="store_false")
        g_common.add_argument("--has_sub_dataset", dest="has_sub_dataset", action="store_true")
        g_common.set_defaults(has_sub_dataset=False)
