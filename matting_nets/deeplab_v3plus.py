import tensorflow as tf
from nets.mobilenet import mobilenet_v2 as mobilenet_v2_slim

slim = tf.contrib.slim


available_extractors = [
    "mobilenet_v2",
]


def build_encoder(network, inputs, is_training, depth_multiplier=None, output_stride=16):
    if network == "mobilenet_v2":
        return mobilenet_v2_slim.mobilenet_base(inputs,
                                                conv_defs=mobilenet_v2_slim.V2_DEF,
                                                depth_multiplier=depth_multiplier,
                                                final_endpoint="layer_18",
                                                output_stride=output_stride,
                                                is_training=is_training)
    else:
        raise NotImplementedError


def get_decoder_end_point(network):
    if network == "mobilenet_v2":
        return "layer_4/depthwise_output"
    else:
        raise NotImplementedError


def aspp(inputs, depth=256, rates=[3, 6, 9]):
    with tf.variable_scope("aspp"):
        branches = []

        with tf.variable_scope("aspp_1x1conv"):
            net = slim.conv2d(inputs, num_outputs=depth, kernel_size=1, stride=1)
            branches.append(net)

        for rate in rates:
            with tf.variable_scope(f"aspp_atrous{rate}"):
                net = slim.separable_conv2d(inputs, num_outputs=None, kernel_size=3, stride=1,
                                            depth_multiplier=1, rate=rate, scope="depthwise_conv")
                net = slim.conv2d(net, num_outputs=depth, kernel_size=1, stride=1, scope="pointwise_conv")
                branches.append(net)

        with tf.variable_scope("aspp_pool"):
            net = tf.reduce_mean(inputs, [1, 2], keep_dims=True, name="global_pool")
            net = slim.conv2d(net, num_outputs=depth, kernel_size=1)
            net = tf.image.resize_bilinear(net, size=inputs.get_shape()[1:3], align_corners=True)
            branches.append(net)

        with tf.variable_scope("concat"):
            concat_logits = tf.concat(branches, 3)
            concat_logits = slim.conv2d(concat_logits, num_outputs=depth, kernel_size=1, stride=1)
            concat_logits = slim.dropout(concat_logits, keep_prob=0.9)

            return concat_logits


def decoder(args, endpoints, aspp, height, width, num_classes):
    with tf.variable_scope("decoder"):
        low_level_feature = endpoints[get_decoder_end_point(args.extractor)]
        net = slim.conv2d(low_level_feature, num_outputs=48, kernel_size=1, scope="feature_projection")
        decoder_features = [net, aspp]

        for i in range(len(decoder_features)):
            decoder_features[i] = tf.image.resize_bilinear(decoder_features[i], size=[height, width],
                                                           align_corners=True)

        net = tf.concat(decoder_features, 3)
        for i in range(2):
            net = slim.separable_conv2d(net, num_outputs=None, kernel_size=3, stride=1,
                                        depth_multiplier=1, scope=f"decoder_conv{i}_depthwise")
            net = slim.conv2d(net, num_outputs=256, kernel_size=1, stride=1, scope=f"decoder_conv{i}_pointwise")

        net = slim.conv2d(net, num_outputs=num_classes, kernel_size=1, stride=1,
                          normalizer_fn=None, activation_fn=None, scope="logits")

        return net


def DeepLab(inputs, args, is_training, depth_multiplier=None, output_stride=16, scope="DeepLab"):
    with tf.variable_scope(scope, "DeepLab", [inputs]):
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            features, endpoints = build_encoder(args.extractor, inputs, is_training, depth_multiplier, output_stride)
            if args.extractor == "mobilenet_v2":
                atrous_rates = []
                result = aspp(features, rates=atrous_rates)
                result = slim.conv2d(result, num_outputs=args.num_classes, kernel_size=1, stride=1,
                                     normalizer_fn=None, activation_fn=None, scope="logits")
            else:
                raise NotImplementedError

            result = tf.image.resize_bilinear(result,
                                              size=[args.height, args.width],
                                              align_corners=True)
    return result, endpoints


def DeepLab_arg_scope(weight_decay=0.00004,
                      batch_norm_decay=0.9997,
                      batch_norm_epsilon=1e-5,
                      batch_norm_scale=True,
                      weights_initializer_stddev=0.09,
                      activation_fn=tf.nn.relu,
                      regularize_depthwise=False,
                      use_fused_batchnorm=True):
    batch_norm_params = {
        "decay": batch_norm_decay,
        "epsilon": batch_norm_epsilon,
        "scale": batch_norm_scale,
        "fused": use_fused_batchnorm,
    }
    if regularize_depthwise:
        depthwise_regularizer = slim.l2_regularizer(weight_decay)
    else:
        depthwise_regularizer = None

    with slim.arg_scope(
            [slim.conv2d, slim.separable_conv2d],
            weights_initializer=tf.truncated_normal_initializer(stddev=weights_initializer_stddev),
            activation_fn=activation_fn,
            normalizer_fn=slim.batch_norm):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope(
                    [slim.conv2d],
                    weights_regularizer=slim.l2_regularizer(weight_decay)):
                with slim.arg_scope(
                        [slim.separable_conv2d],
                        weights_regularizer=depthwise_regularizer) as scope:
                    return scope
