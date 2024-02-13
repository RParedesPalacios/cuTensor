from cuTensor import cuTensor as T

t1=T([4,5,3,2]) # create a 4x5x3x2 tensor on GPU 0 with default name
t1.info()

t2=T([4,5,3,2],device=1) # create a 4x5x3x2 tensor on GPU 1
t2.info()

t3=T([4,5,3,2],name="t3") # create a 4x5x3x2 tensor on GPU 0 with name "t3"
t3.info()

t4=T([4,5,3,2],device=1,name="t4") # create a 4x5x3x2 tensor on GPU 1 with name "t4"
t4.info()


