import cuTensor
from cuTensor import cuTensor as T
import numpy as np

print(cuTensor.__version__)

# Create a tensor
a = T([2,3])
a.fill()

a.print_array()

b = T([2,3])
b.fill()
b.print_array()

c=T.mm(b,a.transpose())
print(c)
c.print_array()


