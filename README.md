
## cuTensor

Provides a python interface to tensors in GPU. 
Personal project just to learn how to manage tensors, how to provide a python interface with pybind etc.

## TODO

Easier installation with just "pip install"

## Install

```console

sudo apt update
sudo apt install g++ gcc-12 g++-12
sudo apt install cmake
sudo apt install nvidia-cuda-toolkit
sudo apt install pybind11-dev 

git clone https://github.com/RParedesPalacios/cuTensor.git

cd cuTensor

# Optional: force CUDA host compiler (recommended on Linux ARM64)
export CUDAHOSTCXX=/usr/bin/g++-12

python setup.py build_ext --inplace 
pip install .

```

If `python setup.py build_ext --inplace` fails on ARM64 with errors in
`/usr/include/aarch64-linux-gnu/bits/math-vector.h`, install `g++-12` and set
`CUDAHOSTCXX=/usr/bin/g++-12` before building.

## Test the installation
```console
python -c "import cuTensor"

```

## See [examples](examples/)
