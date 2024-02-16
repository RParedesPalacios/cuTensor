from cuTensor import cuTensor as T
import numpy as np

# Create a tensor
a = T.from_numpy(np.random.rand(3, 10, 2))
a.print()

b= T.from_numpy(np.random.rand(3, 10, 2))
b.print()

b=b.permute([1,2,0])

c=a*b
c.print()