from cuTensor import cuTensor as T

t1=T([4,5,3,2],0,"t1") # create a 4x5x3x2 tensor on GPU 0 with default name
t1.info()

t1.reshape([6,5,2,2]) # reshape t1 to 6x5x2x2
t1.info()

print(t1.shape) # print the shape of t1
print(t1.dim) # print the dimension of t1
print(t1.device) # print the device of t1
print(t1.size) # print the size of t1

t2=T([3,5,2,2]) 
t2.info()

t3=T.mult2D(t1,t2) # multiply t1 and t2 with broadcasting

t3.info()
