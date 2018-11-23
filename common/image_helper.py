import tensorflow as tf


# Below method is borrowed from tensorflow official repository
# https://github.com/tensorflow/tensorflow/blob/a0b8cee815100b805a24fedfa12b28139d24e7fe/tensorflow/python/ops/image_ops_impl.py
def _verify_compatible_image_shapes(img1, img2):
    """Checks if two image tensors are compatible for applying SSIM or PSNR.
    This function checks if two sets of images have ranks at least 3, and if the
    last three dimensions match.
    Args:
      img1: Tensor containing the first image batch.
      img2: Tensor containing the second image batch.
    Returns:
      A tuple containing: the first tensor shape, the second tensor shape, and a
      list of control_flow_ops.Assert() ops implementing the checks.
    Raises:
      ValueError: When static shape check fails.
    """
    shape1 = img1.get_shape().with_rank_at_least(3)
    shape2 = img2.get_shape().with_rank_at_least(3)
    shape1[-3:].assert_is_compatible_with(shape2[-3:])

    if shape1.ndims is not None and shape2.ndims is not None:
        for dim1, dim2 in zip(reversed(shape1[:-3]), reversed(shape2[:-3])):
            if not (dim1 == 1 or dim2 == 1 or dim1.is_compatible_with(dim2)):
                raise ValueError(
                    "Two images are not compatible: %s and %s" % (shape1, shape2))

    # Now assign shape tensors.
    shape1, shape2 = tf.shape_n([img1, img2])

    checks = []
    checks.append(tf.Assert(
        tf.greater_equal(tf.size(shape1), 3),
        [shape1, shape2], summarize=10))
    checks.append(tf.Assert(
        tf.reduce_all(tf.equal(shape1[-3:], shape2[-3:])),
        [shape1, shape2], summarize=10))
    return shape1, shape2, checks


def sobel_gradient(img):
    """Get image and calculate the result of sobel filter.

    Args:
        imgs: Image Tensor. Either 3-D or 4-D.

    Return:
        A Tensor which concat the result of sobel in both
        horizontally and vertically.
        Therefore, number of the channels is doubled.

    """
    num_channels = img.get_shape().as_list()[-1]

    # load filter which can be reused
    with tf.variable_scope("misc/img_gradient", reuse=tf.AUTO_REUSE):
        filter_x = tf.constant([[-1/8, 0, 1/8],
                                [-2/8, 0, 2/8],
                                [-1/8, 0, 1/8]],
                               name="sobel_x", dtype=tf.float32, shape=[3, 3, 1, 1])
        filter_x = tf.tile(filter_x, [1, 1, num_channels, 1])

        filter_y = tf.constant([[-1/8, -2/8, -1/8],
                                [0, 0, 0],
                                [1/8, 2/8, 1/8]],
                               name="sobel_y", dtype=tf.float32, shape=[3, 3, 1, 1])
        filter_y = tf.tile(filter_y, [1, 1, num_channels, 1])

    # calculate
    grad_x = tf.nn.depthwise_conv2d(img, filter_x,
                                    strides=[1, 1, 1, 1], padding="VALID", name="grad_x")

    grad_y = tf.nn.depthwise_conv2d(img, filter_y,
                                    strides=[1, 1, 1, 1], padding="VALID", name="grad_y")

    grad_xy = tf.concat([grad_x, grad_y], axis=-1)

    return grad_xy


def _first_deriviate_gaussian_filters(size, sigma):
    size = tf.convert_to_tensor(size, tf.int32)
    sigma = tf.convert_to_tensor(sigma, tf.float32)
    sigma2 = tf.square(sigma)

    coords = tf.cast(tf.range(size), sigma.dtype)
    coords -= tf.cast(size - 1, sigma.dtype) / 2.0

    g = tf.square(coords)
    g *= -0.5 / tf.square(sigma)

    g = tf.reshape(g, shape=[1, -1]) + tf.reshape(g, shape=[-1, 1])
    g = tf.reshape(g, shape=[1, -1])  # For tf.nn.softmax().
    g = tf.nn.softmax(g)
    g = tf.reshape(g, shape=[size, size])

    # https://cedar.buffalo.edu/~srihari/CSE555/Normal2.pdf
    # https://github.com/scipy/scipy/blob/v0.14.0/scipy/ndimage/filters.py#L179
    gx = -1 * tf.reshape(coords, shape=[1, -1]) * g / sigma2
    gy = -1 * tf.reshape(coords, shape=[-1, 1]) * g / sigma2

    # gx = tf.reshape(gx, shape=[1, -1])  # For tf.nn.softmax().
    # gy = tf.reshape(gy, shape=[1, -1])  # For tf.nn.softmax().
    # gx = tf.nn.softmax(gx)
    # gy = tf.nn.softmax(gy)

    return tf.reshape(gx, shape=[size, size, 1, 1]), tf.reshape(gy, shape=[size, size, 1, 1])


def first_deriviate_gaussian_gradient(img, sigma):
    """Get image and calculate the result of first deriviate gaussian filter.
    Now, implementation assume that channel is 1.
    https://www.juew.org/publication/CVPR09_evaluation_final_HQ.pdf

    Args:
        imgs: Image Tensor. Either 3-D or 4-D.

    Return:
        A Tensor which concat the result of sobel in both
        horizontally and vertically.
        Therefore, number of the channels is doubled.

    """
    num_channels = img.get_shape().as_list()[-1]
    assert num_channels == 1

    # load filter which can be reused
    with tf.variable_scope("misc/img_gradient", reuse=tf.AUTO_REUSE):
        # truncate for 3 sigma
        half_width = int(3 * sigma + 0.5)
        size = 2 * half_width + 1

        filter_x, filter_y = _first_deriviate_gaussian_filters(size, sigma)

    # calculate
    grad_x = tf.nn.depthwise_conv2d(img, filter_x,
                                    strides=[1, 1, 1, 1], padding="VALID", name="grad_x")

    grad_y = tf.nn.depthwise_conv2d(img, filter_y,
                                    strides=[1, 1, 1, 1], padding="VALID", name="grad_y")

    grad_xy = tf.concat([grad_x, grad_y], axis=-1)
    return grad_xy
