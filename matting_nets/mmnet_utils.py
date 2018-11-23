# Copied and modified the code in tensorflow slim
# https://github.com/tensorflow/tensorflow/blob/r1.6/tensorflow/contrib/layers/python/layers/layers.py

# Implemented quantizable_separable_convolution2d by modifying slim.separable_convoultion: Line 192~243
# Changed operation order for dilated depthwise convolution
# Before: [SpaceToBatchND] -> [DepthwiseConv2dNative] -> [BatchToSpaceND] -> [normalize] -> [activate]
# After : [SpaceToBatchND] -> [DepthwiseConv2dNative] -> [normalize] -> [activate] -> [BatchToSpaceND]

from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.layers import convolutional as convolutional_layers
from tensorflow.python.layers import core as core_layers
from tensorflow.python.layers import normalization as normalization_layers
from tensorflow.python.layers import pooling as pooling_layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.training import moving_averages

DATA_FORMAT_NCHW = 'NCHW'
DATA_FORMAT_NHWC = 'NHWC'
DATA_FORMAT_NCDHW = 'NCDHW'
DATA_FORMAT_NDHWC = 'NDHWC'


@add_arg_scope
def quantizable_separable_convolution2d(
        inputs,
        num_outputs,
        kernel_size,
        depth_multiplier,
        stride=1,
        padding='SAME',
        data_format=DATA_FORMAT_NHWC,
        rate=1,
        activation_fn=nn.relu,
        normalizer_fn=None,
        normalizer_params=None,
        weights_initializer=initializers.xavier_initializer(),
        weights_regularizer=None,
        biases_initializer=init_ops.zeros_initializer(),
        biases_regularizer=None,
        reuse=None,
        variables_collections=None,
        outputs_collections=None,
        trainable=True,
        scope=None):
    """Adds a depth-separable 2D convolution with optional batch_norm layer.
    This op first performs a depthwise convolution that acts separately on
    channels, creating a variable called `depthwise_weights`. If `num_outputs`
    is not None, it adds a pointwise convolution that mixes channels, creating a
    variable called `pointwise_weights`. Then, if `normalizer_fn` is None,
    it adds bias to the result, creating a variable called 'biases', otherwise,
    the `normalizer_fn` is applied. It finally applies an activation function
    to produce the end result.
    Args:
      inputs: A tensor of size [batch_size, height, width, channels].
      num_outputs: The number of pointwise convolution output filters. If is
        None, then we skip the pointwise convolution stage.
      kernel_size: A list of length 2: [kernel_height, kernel_width] of
        of the filters. Can be an int if both values are the same.
      depth_multiplier: The number of depthwise convolution output channels for
        each input channel. The total number of depthwise convolution output
        channels will be equal to `num_filters_in * depth_multiplier`.
      stride: A list of length 2: [stride_height, stride_width], specifying the
        depthwise convolution stride. Can be an int if both strides are the same.
      padding: One of 'VALID' or 'SAME'.
      data_format: A string. `NHWC` (default) and `NCHW` are supported.
      rate: A list of length 2: [rate_height, rate_width], specifying the dilation
        rates for atrous convolution. Can be an int if both rates are the same.
        If any value is larger than one, then both stride values need to be one.
      activation_fn: Activation function. The default value is a ReLU function.
        Explicitly set it to None to skip it and maintain a linear activation.
      normalizer_fn: Normalization function to use instead of `biases`. If
        `normalizer_fn` is provided then `biases_initializer` and
        `biases_regularizer` are ignored and `biases` are not created nor added.
        default set to None for no normalizer function
      normalizer_params: Normalization function parameters.
      weights_initializer: An initializer for the weights.
      weights_regularizer: Optional regularizer for the weights.
      biases_initializer: An initializer for the biases. If None skip biases.
      biases_regularizer: Optional regularizer for the biases.
      reuse: Whether or not the layer and its variables should be reused. To be
        able to reuse the layer scope must be given.
      variables_collections: Optional list of collections for all the variables or
        a dictionary containing a different list of collection per variable.
      outputs_collections: Collection to add the outputs.
      trainable: Whether or not the variables should be trainable or not.
      scope: Optional scope for variable_scope.
    Returns:
      A `Tensor` representing the output of the operation.
    Raises:
      ValueError: If `data_format` is invalid.
    """
    if data_format not in (DATA_FORMAT_NCHW, DATA_FORMAT_NHWC):
        raise ValueError('data_format has to be either NCHW or NHWC.')
    layer_variable_getter = _build_variable_getter({
        'bias': 'biases',
        'depthwise_kernel': 'depthwise_weights',
        'pointwise_kernel': 'pointwise_weights'
    })

    with variable_scope.variable_scope(
            scope,
            'SeparableConv2d', [inputs],
            reuse=reuse,
            custom_getter=layer_variable_getter) as sc:
        inputs = ops.convert_to_tensor(inputs)

        df = ('channels_first'
              if data_format and data_format.startswith('NC') else 'channels_last')
        if num_outputs is not None:
            # Apply separable conv using the SeparableConvolution2D layer.
            layer = convolutional_layers.SeparableConvolution2D(
                filters=num_outputs,
                kernel_size=kernel_size,
                strides=stride,
                padding=padding,
                data_format=df,
                dilation_rate=utils.two_element_tuple(rate),
                activation=None,
                depth_multiplier=depth_multiplier,
                use_bias=not normalizer_fn and biases_initializer,
                depthwise_initializer=weights_initializer,
                pointwise_initializer=weights_initializer,
                bias_initializer=biases_initializer,
                depthwise_regularizer=weights_regularizer,
                pointwise_regularizer=weights_regularizer,
                bias_regularizer=biases_regularizer,
                activity_regularizer=None,
                trainable=trainable,
                name=sc.name,
                dtype=inputs.dtype.base_dtype,
                _scope=sc,
                _reuse=reuse)
            outputs = layer.apply(inputs)

            # Add variables to collections.
            _add_variable_to_collections(layer.depthwise_kernel,
                                         variables_collections, 'weights')
            _add_variable_to_collections(layer.pointwise_kernel,
                                         variables_collections, 'weights')
            if layer.bias is not None:
                _add_variable_to_collections(layer.bias, variables_collections,
                                             'biases')

            if normalizer_fn is not None:
                normalizer_params = normalizer_params or {}
                outputs = normalizer_fn(outputs, **normalizer_params)
        else:
            # Actually apply depthwise conv instead of separable conv.
            dtype = inputs.dtype.base_dtype
            kernel_h, kernel_w = utils.two_element_tuple(kernel_size)
            stride_h, stride_w = utils.two_element_tuple(stride)
            num_filters_in = utils.channel_dimension(
                inputs.get_shape(), df, min_rank=4)
            weights_collections = utils.get_variable_collections(
                variables_collections, 'weights')

            depthwise_shape = [kernel_h, kernel_w, num_filters_in, depth_multiplier]
            depthwise_weights = variables.model_variable(
                'depthwise_weights',
                shape=depthwise_shape,
                dtype=dtype,
                initializer=weights_initializer,
                regularizer=weights_regularizer,
                trainable=trainable,
                collections=weights_collections)
            strides = [1, 1, stride_h,
                       stride_w] if data_format.startswith('NC') else [
                1, stride_h, stride_w, 1
            ]

# DIFFERING PART START

            input = ops.convert_to_tensor(inputs, name="tensor_in")
            filter = ops.convert_to_tensor(depthwise_weights, name="filter_in")
            if rate is None:
                rate = [1, 1]

            def op(input_converted, _, padding):
                outputs = nn_ops.depthwise_conv2d_native(
                    input=input_converted,
                    filter=filter,
                    strides=strides,
                    padding=padding,
                    data_format=data_format)

                num_outputs = depth_multiplier * num_filters_in

                if normalizer_fn is not None:
                    normalizer_params_ = normalizer_params or {}
                    outputs = normalizer_fn(outputs, **normalizer_params_)
                else:
                    if biases_initializer is not None:
                        biases_collections = utils.get_variable_collections(
                            variables_collections, 'biases')
                        biases = variables.model_variable(
                            'biases',
                            shape=[
                                num_outputs,
                            ],
                            dtype=dtype,
                            initializer=biases_initializer,
                            regularizer=biases_regularizer,
                            trainable=trainable,
                            collections=biases_collections)
                        outputs = nn.bias_add(outputs, biases, data_format=data_format)

                if activation_fn is not None:
                    outputs = activation_fn(outputs)

                return outputs

            outputs = nn_ops.with_space_to_batch(
                input=input,
                filter_shape=array_ops.shape(filter),
                dilation_rate=utils.two_element_tuple(rate),
                padding=padding,
                data_format=data_format,
                op=op)

        return utils.collect_named_outputs(outputs_collections, sc.name, outputs)

# DIFFERING PART END


def _build_variable_getter(rename=None):
    """Build a model variable getter that respects scope getter and renames."""

    # VariableScope will nest the getters
    def layer_variable_getter(getter, *args, **kwargs):
        kwargs['rename'] = rename
        return _model_variable_getter(getter, *args, **kwargs)

    return layer_variable_getter


def _model_variable_getter(getter,
                           name,
                           shape=None,
                           dtype=None,
                           initializer=None,
                           regularizer=None,
                           trainable=True,
                           collections=None,
                           caching_device=None,
                           partitioner=None,
                           rename=None,
                           use_resource=None,
                           **_):
    """Getter that uses model_variable for compatibility with core layers."""
    short_name = name.split('/')[-1]
    if rename and short_name in rename:
        name_components = name.split('/')
        name_components[-1] = rename[short_name]
        name = '/'.join(name_components)
    return variables.model_variable(
        name,
        shape=shape,
        dtype=dtype,
        initializer=initializer,
        regularizer=regularizer,
        collections=collections,
        trainable=trainable,
        caching_device=caching_device,
        partitioner=partitioner,
        custom_getter=getter,
        use_resource=use_resource)


def _add_variable_to_collections(variable, collections_set, collections_name):
    """Adds variable (or all its parts) to all collections with that name."""
    collections = utils.get_variable_collections(collections_set,
                                                 collections_name) or []
    variables_list = [variable]
    if isinstance(variable, tf_variables.PartitionedVariable):
        variables_list = [v for v in variable]
    for collection in collections:
        for var in variables_list:
            if var not in ops.get_collection(collection):
                ops.add_to_collection(collection, var)
