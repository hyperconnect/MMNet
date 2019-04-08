## Towards Real-Time Automatic Portrait Matting on Mobile Devices

We tackle the problem of automatic portrait matting on mobile devices.
The proposed model is aimed at attaining real-time inference on mobile devices with minimal degradation of model performance.
Our model MMNet, based on multi-branch dilated convolution with linear bottleneck blocks, outperforms the state-of-the-art model and is orders of magnitude faster.
The model can be accelerated four times to attain 30 FPS on Xiaomi Mi 5 device with moderate increase in the gradient error.
Under the same conditions, our model has an order of magnitude less number of parameters and is faster than Mobile DeepLabv3 while maintaining comparable performance.

<p align="center">
  <img src="https://raw.githubusercontent.com/hyperconnect/MMNet/master/figure/gradient_error_vs_latency.png", width="500", alt="gradient_error_vs_latency">
</p>

The trade-off between gradient error and latency on a mobile device.
Latency is measured using a Qualcomm Snapdragon 820 MSM8996 CPU.
Size of each circle is proportional to the logarithm of the number of parameters used by the model.
Different circles of Mobile DeepLabv3 are created by varying the output stride and width multiplier.
The circles are marked with their width multiplier.
Results using 128 x 128 inputs are marked with * , otherwise, inputs are in 256 x 256.
Notice that MMNet outperforms all other models forming a Pareto front.
The number of parameters for LDN+FB is not reported in their paper.


## Requirements

- Python 3.6+
- Tensorflow 1.6

## Installation

```
git clone --recursive https://github.com/hyperconnect/MMNet.git
pip3 install -r requirements/py36-gpu.txt
```

## Dataset
Dataset for training and evaluation has to follow directory structure as depticted below.
To use other name than `train` and `test`, one can utilize `--dataset_split_name` argument in *train.py* or *evaluate.py*.
```
dataset_directory
  |___ train
  |   |__ mask
  |   |__ image
  |
  |___ test
      |__ mask
      |__ image
```


## Training
In `scripts` directory, you can find example scripts for training and evaluation of MMNet and Mobile DeepLabv3.
Training scripts accept two arguments: `dataset path` and `train directory`.
`dataset path` has to point to directory with structure described in the previous section.

### MMNet
Training of MMNet with depth multiplier 1.0 and input image size 256.

```bash
./scripts/train_mmnet_dm1.0_256.sh /path/to/dataset /path/to/training/directory
```

### Mobile DeepLabv3
Training of Mobile DeepLabv3 with output stride 16, depth multiplier 0.5 and input image size 256.

```bash
./scripts/train_deeplab_os16_dm0.5_256.sh /path/to/dataset /path/to/training/directory
```



## Evaluation
Evaluation scripts, same as training scripts, accept two arguments: `dataset path` and `train directory`.
If `train directory` argument points to specific checkpoint file, only that checkpoint file will be evaluated, otherwise the latest checkpoint file will be evaluated.
It is recommended to run evaluation scripts together with training scripts in order to get evaluation metrics for every checkpoint file.

### MMNet

```bash
./scripts/valid_mmnet_dm1.0_256.sh /path/to/dataset /path/to/training/directory
```

### Mobile DeepLabv3

```bash
./scripts/valid_deeplab_os16_dm0.5_256.sh /path/to/dataset /path/to/training/directory
```

## Demo

Refer to `demo/demo.mp4`.

## License

[Apache License 2.0](LICENSE)
