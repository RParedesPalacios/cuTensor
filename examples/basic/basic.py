import cuTensor 
from cuTensor import cuTensor as T

cuTensor.hw_info() # print hardware information

t1=T([2,3,5,4],0,"t1") # create a 4x5x3x2 tensor on GPU 0 with default name
t1.info()

t1.fill() # fill the tensor with 1.0
t1.print() # print the tensor

t1.permute([2,3,0,1]) # permute the tensor
t1.print() # print the tensor

t1.permute([2,3,0,1]) # permute the tensor
t1.print() # print the tensor