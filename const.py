import tensorflow as tf


TF_SESSION_CONFIG = tf.ConfigProto(
    gpu_options=tf.GPUOptions(allow_growth=True),
    log_device_placement=False,
    device_count={"GPU": 1})

CKPT_EXTENSION = ".my-ckpt"
