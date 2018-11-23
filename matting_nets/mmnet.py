import tensorflow as tf

from matting_nets.mmnet_utils import quantizable_separable_convolution2d

slim = tf.contrib.slim
separable_conv = quantizable_separable_convolution2d


def multiply_depth(depth, depth_multiplier, min_depth=8, divisor=8):
    multiplied_depth = round(depth * depth_multiplier)
    divisible_depth = (multiplied_depth + divisor // 2) // divisor * divisor
    return max(min_depth, divisible_depth)


def init_block(inputs, depth, depth_multiplier, name):
    depth = multiply_depth(depth, depth_multiplier)
    with tf.variable_scope(name):
        net = slim.conv2d(inputs, num_outputs=depth, kernel_size=3, stride=2, scope="conv")
    return net


def encoder_block(inputs, expanded_depth, output_depth, depth_multiplier, rates, stride, name,
                  activation_fn=tf.identity):
    expanded_depth = multiply_depth(expanded_depth, depth_multiplier)
    output_depth = multiply_depth(output_depth, depth_multiplier)

    with tf.variable_scope(name):
        convs = []
        for i, rate in enumerate(rates):
            with tf.variable_scope(f"branch{i}"):
                conv = slim.conv2d(inputs, num_outputs=expanded_depth, kernel_size=1, stride=1, scope="pointwise_conv")
                if stride > 1:
                    conv = separable_conv(conv, num_outputs=None, kernel_size=3, stride=stride, depth_multiplier=1,
                                          scope="depthwise_conv_stride")
                conv = separable_conv(conv, num_outputs=None, kernel_size=3, stride=1, depth_multiplier=1,
                                      rate=rate, scope="depthwise_conv_dilation")
                convs.append(conv)

        with tf.variable_scope("merge"):
            if len(convs) > 1:
                net = tf.concat(convs, axis=-1)
            else:
                net = convs[0]

            net = slim.conv2d(net, num_outputs=output_depth, kernel_size=1, stride=1, activation_fn=activation_fn,
                              scope="pointwise_conv")
            net = slim.dropout(net, scope="dropout")

    return net


def decoder_block(inputs, shortcut_input, compressed_depth, shortcut_depth, depth_multiplier, num_of_resize, name):
    compressed_depth = multiply_depth(compressed_depth, depth_multiplier)
    if shortcut_depth is not None:
        shortcut_depth = multiply_depth(shortcut_depth, depth_multiplier)

    with tf.variable_scope(name):
        net = slim.conv2d(inputs, num_outputs=compressed_depth, kernel_size=1, stride=1, scope="pointwise_conv")

        for i in range(num_of_resize):
            resize_shape = [v * 2**(i+1) for v in inputs.get_shape().as_list()[1:3]]
            net = tf.image.resize_bilinear(net, size=resize_shape, name=f"resize_bilinear_{i}")

        if shortcut_input is not None:
            with tf.variable_scope("shortcut"):
                shortcut = separable_conv(shortcut_input, num_outputs=None, kernel_size=3,
                                          stride=1, depth_multiplier=1, rate=1, scope="depthwise_conv")
                shortcut = slim.conv2d(shortcut, num_outputs=shortcut_depth, kernel_size=1, stride=1,
                                       scope="pointwise_conv")
                net = tf.concat([net, shortcut], axis=-1, name="concat")

    return net


def final_block(inputs, num_outputs, name):
    with tf.variable_scope(name):
        net = slim.conv2d(inputs, num_outputs=num_outputs, kernel_size=1, stride=1,
                          activation_fn=None, normalizer_fn=None, scope="pointwise_conv")
    return net


def MMNet(inputs, is_training, depth_multiplier=1.0, scope="MMNet"):
    endpoints = {}
    with tf.variable_scope(scope, "MMNet", [inputs]):
        with slim.arg_scope([slim.batch_norm], is_training=is_training, activation_fn=None):
            with slim.arg_scope([slim.dropout], is_training=is_training):
                endpoints["init_block"] = init_block(inputs, 32, depth_multiplier, "init_block")

                # encoder do downsampling
                endpoints["enc_block0"] = encoder_block(endpoints["init_block"], 16, 16, depth_multiplier,
                                                        [1, 2, 4, 8], 2, "enc_block0")
                endpoints["enc_block1"] = encoder_block(endpoints["enc_block0"], 16, 24, depth_multiplier,
                                                        [1, 2, 4, 8], 1, "enc_block1")
                endpoints["enc_block2"] = encoder_block(endpoints["enc_block1"], 24, 24, depth_multiplier,
                                                        [1, 2, 4, 8], 1, "enc_block2")
                endpoints["enc_block3"] = encoder_block(endpoints["enc_block2"], 24, 24, depth_multiplier,
                                                        [1, 2, 4, 8], 1, "enc_block3")
                endpoints["enc_block4"] = encoder_block(endpoints["enc_block3"], 32, 40, depth_multiplier,
                                                        [1, 2, 4], 2, "enc_block4")
                endpoints["enc_block5"] = encoder_block(endpoints["enc_block4"], 64, 40, depth_multiplier,
                                                        [1, 2, 4], 1, "enc_block5")
                endpoints["enc_block6"] = encoder_block(endpoints["enc_block5"], 64, 40, depth_multiplier,
                                                        [1, 2, 4], 1, "enc_block6")
                endpoints["enc_block7"] = encoder_block(endpoints["enc_block6"], 64, 40, depth_multiplier,
                                                        [1, 2, 4], 1, "enc_block7")
                endpoints["enc_block8"] = encoder_block(endpoints["enc_block7"], 80, 80, depth_multiplier,
                                                        [1, 2], 2, "enc_block8")
                endpoints["enc_block9"] = encoder_block(endpoints["enc_block8"], 120, 80, depth_multiplier,
                                                        [1, 2], 1, "enc_block9", tf.nn.relu6)

                endpoints["dec_block0"] = decoder_block(endpoints["enc_block9"], endpoints["enc_block4"],
                                                        64, 64, depth_multiplier, 1, "dec_block0")
                endpoints["dec_block1"] = decoder_block(endpoints["dec_block0"], endpoints["enc_block0"],
                                                        40, 40, depth_multiplier, 1, "dec_block1")

                endpoints["dec_block2"] = encoder_block(endpoints["dec_block1"], 40, 40, depth_multiplier,
                                                        [1, 2, 4], 1, "dec_block2")
                endpoints["dec_block3"] = encoder_block(endpoints["dec_block2"], 40, 40, depth_multiplier,
                                                        [1, 2, 4], 1, "dec_block3")

                endpoints["dec_block4"] = decoder_block(endpoints["dec_block3"], None,
                                                        16, None, depth_multiplier, 2, "dec_block4")

                # Final Deconvolution
                endpoints["final_block"] = final_block(endpoints["dec_block4"], num_outputs=2, name="final_block")

                # aux output
                endpoints["aux_block"] = final_block(endpoints["enc_block9"], num_outputs=2, name="aux_block")

    return endpoints["final_block"], endpoints


def MMNet_arg_scope(use_fused_batchnorm=True,
                       regularize_depthwise=False,
                       weight_decay=0.0,
                       dropout=0.0):
    if regularize_depthwise and weight_decay != 0.0:
        depthwise_regularizer = slim.l2_regularizer(weight_decay)
    else:
        depthwise_regularizer = None

    with slim.arg_scope([slim.conv2d, separable_conv], activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm):
        with slim.arg_scope([slim.batch_norm], fused=use_fused_batchnorm, scale=True):
            with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(weight_decay)):
                with slim.arg_scope([separable_conv], weights_regularizer=depthwise_regularizer):
                    with slim.arg_scope([slim.dropout], keep_prob=1 - dropout) as scope:
                        return scope
