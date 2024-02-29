from cuTensor import cuTensor as T
import numpy as np


# create a tensor from numpy array
b = T.from_array(np.random.rand(2,3,5), name="b")
print(b)

# reshape tensor
b.reshape([6,5])  # in-place operation
print(b)

# transpose tensor, returns a new tensor
b=b.transpose()
print(b)

# or
b=b.permute([1,0]) #returns a new tensor

# or 
b=~b #returns a new tensor


# reshape tensor
b.reshape([2,3,5])
print(b)

#permute tensor
b=b.permute([2,0,1])
print(b)


