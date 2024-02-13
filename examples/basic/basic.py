from cuTensor import cuTensor as T

t1=T([4,5,3,2]) # create a 4x5x3x2 tensor on GPU 0 with default name
t1.info()

t2=T([3,5,2,2]) 
t2.info()

t3=T.mult2D(t1,t2) # multiply t1 and t2 with broadcasting

t3.info()
