
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

