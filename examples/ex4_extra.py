from cuTensor import cuTensor as T
import numpy as np


def transform(data):
    data[:]*=2
    # or any complex transformation...


b = T([2,3],name="Tensor b")
b.fill(1)

print(b)
b.print_array()

b.apply(transform) # apply function on CPU and copy back to GPU
b.print_array()


# pass more iformation to the function:
def transform2(data,size):
    data[:size//2]*=2

b.apply(transform2,b.size) 
b.print_array()

