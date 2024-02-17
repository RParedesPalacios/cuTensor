import cuTensor
from cuTensor import cuTensor as T
import numpy as np

cuTensor.hw_info()

print(cuTensor.__version__)

# Create a tensor
a = T.from_array(np.random.rand(2,3), name="a")
a.print_array()

b = T.from_array(np.random.rand(2,3), name="b")
b.print_array()

c=T.mm(b,a.transpose())
c.setName("c")
print(c)
c.print_array()


