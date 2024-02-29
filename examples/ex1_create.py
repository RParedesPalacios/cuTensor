import cuTensor
from cuTensor import cuTensor as T
import numpy as np

# get Hw info
cuTensor.hw_info()

# get cuTensor version
print(cuTensor.__version__)

# create a tensor from shape
a = T([2,3])

# create a tensor from shape and name
a = T([2,3], name="a")

# create a tensor from shape, name and device
a = T([2,3], name="a", device=0)

# print info of tensor 
print(a)

# or
a.print()

# print array values of tensor
a.print_array()

# fill tensor with some value
a.fill(1.0)
a.print_array()

# create a tensor from numpy array
b = T.from_array(np.random.rand(2,3), name="b")
b.print_array()

# create a tensor from numpy array and device
b = T.from_array(np.random.rand(2,3), name="b", device=0)

# clone tensor
c = b.clone("c")
print(c)

# getname properties of tensor
print(b.name)
print(b.device)
print(b.shape)
print(b.size)
print(b.stride)
print(b.dim)




