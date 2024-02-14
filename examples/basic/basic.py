import cuTensor 
from cuTensor import cuTensor as T
import numpy as np

cuTensor.hw_info() # print hardware information

def norm(data):
    norm_value = np.linalg.norm(data)
    data[:] = data / norm_value

# Create a cuTensor object
t = T([4, 5, 2])
t.fill()

# Apply the function to the tensor
# from GPU to CPU and back to GPU
t.apply(norm)
t.print()