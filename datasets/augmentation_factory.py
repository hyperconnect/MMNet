from math import pi
from types import SimpleNamespace

import tensorflow as tf


_available_augmentation_methods_TF = [
    "resize_random_scale_crop_flip_rotate",
    "resize_bilinear",
]
_available_augmentation_methods = (
    _available_augmentation_methods_TF +
    [
        "no_augmentation",
    ]
)


def expand_squeeze(aug_fun):
    def fun(image, output_height: int, output_width: int, channels, **kwargs):
        image = tf.expand_dims(image, 0)
        image = aug_fun(image, output_height, output_width, channels, **kwargs)
        return tf.squeeze(image)
    return fun


@expand_squeeze
def resize_bilinear(image, output_height: int, output_width: int, channels=3, **kwargs):
    # Resize the image to the specified height and width.
    image = tf.image.resize_bilinear(image, (output_height, output_width), align_corners=True)
    return image


def _generate_rand(min_factor, max_factor, step_size):
    """Gets a random value.
         Args:
            min_factor: Minimum value.
            max_factor: Maximum value.
            step_size: The step size from minimum to maximum value.
         Returns:
            A random value selected between minimum and maximum value.
         Raises:
            ValueError: min_factor has unexpected value.
    """
    if min_factor < 0 or min_factor > max_factor:
        raise ValueError("Unexpected value of min_factor.")
    if min_factor == max_factor:
        return tf.to_float(min_factor)
        # When step_size = 0, we sample the value uniformly from [min, max).
    if step_size == 0:
        return tf.random_uniform([1],
                                 minval=min_factor,
                                 maxval=max_factor)
        # When step_size != 0, we randomly select one discrete value from [min, max].
    num_steps = int((max_factor - min_factor) / step_size + 1)
    scale_factors = tf.lin_space(min_factor, max_factor, num_steps)
    shuffled_scale_factors = tf.random_shuffle(scale_factors)
    return shuffled_scale_factors[0]


def _scale_image(image, scale_height=1.0, scale_width=1.0):
    """Scales image.
         Args:
            image: Image with shape [height, width, 3].
            scale_height: The value to scale height of image.
            scale_width: The value to scale of width image.
         Returns:
            Scaled image.
    """
    # No random scaling if scale == 1.
    if scale_height == 1.0 and scale_width == 1.0:
        return image

    image_shape = tf.shape(image)
    new_dim = tf.to_int32([tf.to_float(image_shape[0]) * scale_height, tf.to_float(image_shape[1]) * scale_width])
    # Need squeeze and expand_dims because image interpolation takes
    # 4D tensors as input.
    image = tf.squeeze(tf.image.resize_bilinear(
        tf.expand_dims(image, 0),
        new_dim,
        align_corners=True), [0])

    return image


def random_crop_flip(image, output_height: int, output_width: int, channels=3, **kwargs):
    image = tf.random_crop(image, (output_height, output_width, channels))
    image = tf.image.random_flip_left_right(image)
    return image


def rotate_with_crop(image, args):
    if args.rotation_range == 0:
        return image

    rotation_amount_degree = tf.random_uniform(
        shape=[],
        minval=-args.rotation_range,
        maxval=args.rotation_range,
    )
    rotation_amount_radian = rotation_amount_degree * pi / 180.0
    image = tf.contrib.image.rotate(image, rotation_amount_radian)

    height_and_width = tf.shape(image)
    height, width = height_and_width[0], height_and_width[1]
    max_min_ratio = tf.cast(
        tf.maximum(height, width) / tf.minimum(height, width),
        dtype=tf.float32,
    )
    coef_inv = tf.abs(tf.cos(rotation_amount_radian)) + max_min_ratio * tf.abs(tf.sin(rotation_amount_radian))
    height_crop = tf.cast(tf.round(tf.cast(height, dtype=tf.float32) / coef_inv), dtype=tf.int32)
    width_crop = tf.cast(tf.round(tf.cast(width, dtype=tf.float32) / coef_inv), dtype=tf.int32)
    image = tf.image.crop_to_bounding_box(
        image,
        offset_height=(height - height_crop) // 2,
        offset_width=(width - width_crop) // 2,
        target_height=height_crop,
        target_width=width_crop,
    )
    image = tf.image.resize_bilinear([image], (height, width))[0]
    return image


@expand_squeeze
def resize_bilinear(image, output_height: int, output_width: int, channels=3, **kwargs):
    # Resize the image to the specified height and width.
    image = tf.image.resize_bilinear(image, (output_height, output_width), align_corners=True)
    return image


def resize_random_scale_crop_flip_rotate(image, output_height: int, output_width: int, channels=3, **kwargs):
    image = resize_bilinear(image, output_height, output_width, channels=channels)

    scale_height = _generate_rand(1.0, 1.15, 0.01)
    scale_width = _generate_rand(1.0, 1.15, 0.01)
    image = _scale_image(image, scale_height, scale_width)
    image = random_crop_flip(image, output_height, output_width, channels)

    do_rotate = tf.less(tf.random_uniform([], minval=0, maxval=1), 0.5)
    image = tf.cond(
        pred=do_rotate,
        true_fn=lambda: rotate_with_crop(image, SimpleNamespace(rotation_range=15)),
        false_fn=lambda: image,
    )

    return image


def get_augmentation_fn(name):
    """Returns augmentation_fn(image, height, width, channels, **kwargs).
    Args:
        name: The name of the preprocessing function.
        Returns:
        augmentation_fn: A function that preprocessing a single image (pre-batch).
            It has the following signature:
                image = augmentation_fn(image, output_height, output_width, ...).

    Raises:
        ValueError: If Preprocessing `name` is not recognized.
    """
    if name not in _available_augmentation_methods:
        raise ValueError(f"Augmentation name [{name}] was not recognized")

    def augmentation_fn(image, output_height: int, output_width: int, channels, **kwargs):
        return eval(name)(
            image, output_height, output_width, channels, **kwargs)

    return augmentation_fn
