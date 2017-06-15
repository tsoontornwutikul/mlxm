# mlxm: Machine Learning Experiment Manager

**mlxm** is a simple command line-based Python framework designed to help you run, manage and perform analyses on machine learning experiments in a unified manner.

*This project is a work in progress. Expect subtle bugs and missing functionalities.*


## Installation

### Dependencies

This framework is tested on Python 3.5, with the following package dependencies:
 
 - numpy
 - retrying
 
To use any part of the framework under the `keras` module, you will also need:

 - Keras 2
 - Tensorflow 1.0+
 
### Setup

The simplest way is to simply clone the entire repository as a submodule into a subdirectory named `mlxm` of whatever project you are working on.


## Examples

### Preparation

Make sure you are in your project's directory where the `mlxm` directory is located in.  

### Train a simple fully-connected feed-forward neural network classifier for MNIST

```shell
$ python -m mlxm.keras.experiments.train_classifier name=test-mnist model=mnist.fc dataset=mnist dataset.flatten=1
```

Under the hood, a neural network for MNIST (defined in `mlxm.keras.models.mnist.fc`) is created with default parameters and is trained on the MNIST dataset (defined in `mlxm.keras.datasets.mnist`) with flattened inputs. All data related to training are saved under `./output/experiment/mnist.fc/test/`. Saved data include the parameters used, the model definition in YAML format, the training loss and error history, and the final model after training. Tensorboard data are also automatically saved under `./output/experiment-tb/mnist.fc/test/for-readme@<timestamp>`.

Note that if you try to run a new experiment under the same name for the same model, the program will simply exit without doing anything. This is to avoid accidentally replacing previous results. To override this behavior, append `forced=1` to the end of the command.

### Running it 5 times

```shell
$ python -m mlxm.experiments.run_multiple n=5 experiment=mlxm.keras.experiments.train_classifier name=test model=mnist.fc dataset=mnist dataset.flatten=1
```

This will run 10 independent experiments sequentially under `./output/experiment/mnist.fc/+test/1` through `./output/experiment/mnist.fc/+test/5`.

### Running a 5-fold cross-validation

```shell
$ python -m mlxm.experiments.run_multiple n=5 experiment=mlxm.keras.experiments.train_classifier name=test model=mnist.fc dataset=mnist dataset.flatten=1 dataset.cv.folds=5
```

### Running different variations (i.e. hyperparameter search)

```shell
$ python -m mlxm.experiments.run_variations experiment=mlxm.keras.experiments.train_classifier name=test model=mnist.fc ~model.dropout=0/0.2/0.5 dataset=mnist dataset.flatten=1 optimizer=adam ~optimizer.adam.lr=0.1/0.01/1e-3/1e-4/1e-5
```

This will run a total of 15 (3 times 5) different combinations of learning rate and Dropout settings.


## Using your own model, dataset, training algorithm, etc.

To be written.


## Notes

This project was originally created for personal research uses and thus the code is still not fully refined nor well-documented. It may (actually does) have bugs that may interfere with the progress of your experiments. Please use it at your own risk.

### Why not standard command line argument format?

We found it easier to type and skim through the command this way:

```shell
$ python -m mlxm.keras.experiments.train_classifier name=test model=mnist.fc model.n_units=1000,1000,1000 model.activation=elu model.L2.weight=1e-6 model.dropinput=0.5 model.dropout=0.5 model.batchnorm=1 dataset=mnist dataset.flatten=1 dataset.n_train=50000 dataset.n_valid=10000 optimizer=adam optimizer.adam.lr=0.001 optimizer.adam.beta1=0.5 history.enabled=1 valid.every=1000 train.batch_size=128 train.batches=100000 path.result.main.base=/results path.result.tensorboard.base=/results-tensorboard
```

This also allows values to have a dash (-) prefix (e.g. a negative value).

### Why not just use \<insert name of another framework\>?

We started with [pylearn2](https://github.com/lisa-lab/pylearn2) only to watch it getting slowly abandoned. The other frameworks that were available at that time we started this project did not match our requirements in a way that it was worth (partially) reinventing the wheel. In any case, please check out other frameworks/libraries to see if any of them matches your requirements better. Some of them are listed below, in no particular order:

 - [kur](https://github.com/deepgram/kur)
 - [MXNet](https://github.com/dmlc/mxnet)
 - [Caffe](https://github.com/BVLC/caffe)
 - [Caffe2](https://github.com/caffe2/caffe2)
 - [Torch](https://github.com/torch/torch7)
 - [PyTorch](https://github.com/pytorch/pytorch)
 - [Lasagne](https://github.com/Lasagne/Lasagne)
 
 
Alternatively, you can simply use the underlying [Keras](https://github.com/fchollet/keras) and/or [Tensorflow](https://github.com/tensorflow/tensorflow) directly.


## License

This project is licensed under the [MIT license](LICENSE).
