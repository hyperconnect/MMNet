import random
from pathlib import Path

import tensorflow as tf
from termcolor import colored

from datasets.data_wrapper_base import DataWrapperBase


class MattingDataWrapper(DataWrapperBase):
    """ MattingDataWrapper
    dataset_dir
    |___name
        |__mask
        |__image
    """
    def __init__(
        self,
        args,
        session,
        dataset_split_name: str,
        is_training: bool=False,
        name: str="MattingDataWrapper",
    ):
        super().__init__(args, dataset_split_name, is_training, name=name)

        self.image_channels = 3

        self.IMAGE_DIR_NAME = "image"
        self.MASK_DIR_NAME = "mask"

        self.image_label_dirs, self.mask_label_dirs = self.build_dataset_paths()

        self.setup()

        self.setup_dataset((self.image_placeholder, self.mask_placeholder))
        self.setup_iterator(
            session,
            (self.image_placeholder, self.mask_placeholder),
            (self.image_fullpathes, self.mask_fullpathes),
        )

    def setup(self):
        self.image_fullpathes = []
        self.mask_fullpathes = []

        for image in sorted(self.get_all_images(self.image_label_dirs)):
            imagepath = str(image)
            self.image_fullpathes.append(imagepath)
            maskpath = imagepath.replace(f"/{self.IMAGE_DIR_NAME}/", f"/{self.MASK_DIR_NAME}/")
            assert Path(maskpath).exists(), f"{maskpath} not found"
            self.mask_fullpathes.append(maskpath)

        self._num_samples = len(self.image_fullpathes)
        self.log.info(colored(f"Num Data: {self._num_samples}", "red"))

        if self.shuffle:
            shuffled_data = list(zip(self.image_fullpathes, self.mask_fullpathes))
            random.shuffle(shuffled_data)

            self.image_fullpathes, self.mask_fullpathes = zip(*shuffled_data)
            self.log.info(colored("Data shuffled!", "red"))

        self.image_placeholder = tf.placeholder(tf.string, len(self.image_fullpathes))
        self.mask_placeholder = tf.placeholder(tf.string, len(self.mask_fullpathes))

    @property
    def num_samples(self):
        return self._num_samples

    @property
    def padded_original_image_dummy_shape(self):
        return (1, 1, 4)

    def build_dataset_paths(self):
        dataset_paths = self.get_all_dataset_paths()

        image_label_dirs = []
        mask_label_dirs = []
        for dataset_path in dataset_paths:
            image_dataset_dir = dataset_path / self.IMAGE_DIR_NAME
            mask_dataset_dir = dataset_path / self.MASK_DIR_NAME

            image_label_dir = image_dataset_dir
            mask_label_dir = mask_dataset_dir

            image_label_dirs.append(image_label_dir)
            mask_label_dirs.append(mask_label_dir)

        return image_label_dirs, mask_label_dirs

    def _parse_function(self, imagename, maskname):
        image = tf.image.decode_jpeg(tf.read_file(imagename), channels=self.image_channels)
        if self.need_label:
            mask = tf.image.decode_jpeg(tf.read_file(maskname), channels=1)
        else:
            mask = tf.zeros([self.args.height, self.args.width])

        comb = tf.concat([mask, image], axis=2)  # Concat trick for syncronizing locations of random crop

        comb_original = self.resize_and_padding_before_augmentation(comb, [self.args.height, self.args.width])
        comb_augmented = self.augment_image(comb, self.args, channels=self.image_channels+1)

        if self.args.target_eval_shape:
            mask_original = tf.image.resize_images(
                tf.cast(mask, tf.float32) / 255.0,
                self.args.target_eval_shape,
            )
        else:
            mask_original = tf.squeeze(comb_original[:, :, 0:1], 2)  # mask_original = comb_original[:, :, 0]

        image_original = comb_original[:, :, 1:]

        mask_augmented, image_augmented = comb_augmented[:, :, 0:1], comb_augmented[:, :, 1:]
        mask_augmented = tf.reshape(mask_augmented, [self.args.height, self.args.width]) / 255.0
        image_augmented = tf.reshape(
            image_augmented,
            [self.args.height, self.args.width, self.image_channels],
        )

        return image_original, mask_original, image_augmented, mask_augmented

    @staticmethod
    def add_arguments(parser):
        pass
