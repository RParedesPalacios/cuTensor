
## Install

```console
sudo apt install nvidia-cuda-toolkit
pip install pybind11

git clone https://github.com/RParedesPalacios/cuTensor.git

cd cuTensor

make
```

## BUILD

```console
python setup.py build_ext --inplace 
python setup.py sdist bdist_wheel
pip install .
```