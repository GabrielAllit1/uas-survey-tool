import pycuda.driver as cuda
import pycuda.autoinit
print(cuda.Device.count())  # Should print the number of GPUs