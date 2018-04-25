# PyTorch
<p align="center"><img width="40%" src="https://raw.githubusercontent.com/pytorch/pytorch/master/docs/source/_static/img/pytorch-logo-dark.png" /></p>


PyTorch is more than a simple Deep Learning Framework, it is a Python package that provides two high-level features:

* Tensor computation (like NumPy) with strong GPU acceleration
* Deep neural networks built on a tape-based autograd system

## Installation
To install PyTorch, it is recommended to use Conda package manager, as it follows:

```bash
conda install pytorch cuda80 torchvision -c soumith
```

Right now, conda packages are only available on Linux and OSX. Sorry Windows users :(

## Why PyTorch (Propaganda time)

### Python First
Most current frameworks like Caffe or Tensorflow define tools and bindings around static C/C++ routines, sometimes in a Un-Pythonic way. PyTorch aims to put Python first and offer a simple and extendible syntax, such that is not differentiable from that of classical libraries such as Numpy/Scipy or scikit-learn. This implies that is also possible to extend PyTorch by Cython means.

### More than a Neural Network library
PyTorch is a general tensor/matrix library that allows to perform operations on a GPU without any hassle nor any
complicated syntax. It is just as simple as invoking the ``.cuda()/.cpu()`` function, and all your operations can reside on the GPU or the CPU transparently. PyTorch is more close to a GPU Numpy, allowing to perform calculations beyond the Deep Learning realm.

Actually, PyTorch integrates very well with several high performance computation libraries, such as Intel MKL and NVIDIA cuDNN.

**PS:** PyTorch is more fast than a Chinese veto on a North Korean sanction resoultion at the UNSC.

## Debug as you execute
Because PyTorch instructions can be executed directly on a Python interpreter, you can call each instruction and see its result synchronously, differring from other asynchronous frameworks on which you must compile a model and then execute it to see if it is working properly. Say goodbye to execution engines and sessions.

## Autograd
With respect to Deep Learning applications, PyTorch defines a backpropagation graph on which each node represents a mathematical operation, whereas the edges represent the sequence and forward relation between them. Different from TensorFlow, PyTorch defines dynamic graphs defined at run-time rather than at compile-time, allowing to change a networks' architecture easily and with minimal time overhead.

PyTorch uses a technique called reverse-mode auto-differentiation, which allows you to change the way your network behaves arbitrarily with zero lag or overhead. It's state-of-the-art performance comes from several research papers on this topic, as well as current and past work such as autograd, Chainer, etc. This is actually one of the most efficient and fast current implementations so far.

<p align=center><img width="80%" src="https://raw.githubusercontent.com/pytorch/pytorch/master/docs/source/_static/img/dynamic_graph.gif" /></p>

## Extensions and tools
Along with equivalent and interoperable numpy Linear Algebra and scipy Mathematic operations, PyTorch offers a great ecosystem of NN layers, optimizers and data loading functions.

## Acknowledgments
Some of the basic tutorial examples are based on [jcjohnson's PyTorch basic examples](https://github.com/jcjohnson/pytorch-examples). Other sources are taken directly from PyTorch and TorchVision documentation.