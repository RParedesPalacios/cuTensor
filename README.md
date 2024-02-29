
## cuTensor

Provides an python interface to tensors in GPU. 
Personal project just to learn how to manage tensors, how to provide a python interface with pybind etc.

## TODO

Easier installation with just "pip install"

## Install

```console

sudo apt install g++
sudo apt install cmake
sudo apt install nvidia-cuda-toolkit
sudo apt install pybind11-dev 

git clone https://github.com/RParedesPalacios/cuTensor.git

cd cuTensor

python setup.py build_ext --inplace 
pip install .

```

## Test the installation
```console
python -c "import cuTensor"

```

## See [examples](examples/)
