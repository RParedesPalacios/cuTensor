from cuTensor import cuTensor as T
import numpy as np


# create a tensor from numpy array
b = T.from_numpy(np.random.rand(2,3,5), name="b")
print(b)

a=T.from_numpy(np.random.rand(2,3,5), name="a")
print(a)

# sum of tensors
c=a+b
print(c)

# can be broadcasted, shape of the larger tensor is preserved
a=T.from_numpy(np.random.rand(4,2,3,5), name="a")

c=a+b
c.setName("c") # we can set name of tensor
print(c)

print("==========")
c=b+a
c.setName("c")
print(c)

# with scalar
c=a+1

# substraction
c=a-b
c=a-1  

# multiplication, elementwise product
c=a*b

c=2*a

# division, elementwise division
c=a/b # the same:
c=a*(1/b)

c=a/2
c=2/a

# power
c=a**2

# len
print(len(a))

# ...
c=(2*a+b)/-(b**2)

# matrix multiplication (mult2D), broadcasting is supported
a=T.from_numpy(np.random.rand(2,10,3,5), name="a")
b=T.from_numpy(np.random.rand(3,5,4,7), name="b")

c=T.mm(a,b) # 2,10,3,5 * 3,5,4,7 = 2,10,4,7
print(c)

# transpose, etc...
a=T.from_numpy(np.random.rand(3,5), name="a")
b=T.from_numpy(np.random.rand(3,5), name="b")

c=T.mm(a,~b) # 3,5 * 5,3 = 3,3, b is not modified, a new tensor is created in ~b
print(c)

# OR
c=a@~b
print(c)
