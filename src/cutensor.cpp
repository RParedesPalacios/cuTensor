#include <iostream>
#include <iomanip>
#include "cutensor.h"
#include "gpu.h"

using namespace std;

////////////////////////////////////////
// Constructor and Destructor
////////////////////////////////////////
cuTensor::cuTensor()
{
    ndim = 0;
    size = 0;
    shape = {};
    ptr = nullptr;
}
cuTensor::cuTensor(const vector<int> &s, const int dev, float *cpu_ptr)
{
    ndim = s.size();
    shape = s;
    device = dev;
    
    size=1;
    for (int i = ndim - 1; i >= 0; i--) {size *= shape[i];}
        
    ptr = gpu_create_tensor(device,size);
    if (cpu_ptr != nullptr)
        gpu_copy_to_device(device, size, ptr, cpu_ptr);

}

cuTensor::cuTensor(const vector<int> &shape):cuTensor(shape,0,nullptr){}
cuTensor::cuTensor(const vector<int> &shape, float *cpu_ptr):cuTensor(shape,0,cpu_ptr){}
cuTensor::cuTensor(const vector<int> &shape, const int dev):cuTensor(shape,dev, nullptr){}

cuTensor::~cuTensor()
{
    gpu_delete_tensor(device,ptr);
}

////////////////////////////////////////
// Methods
////////////////////////////////////////
void cuTensor::fill(float value)
{
    gpu_fill_(device, size, ptr, value);
}
void cuTensor::print()
{
    cout << "Tensor in device: GPU " << device << endl;
    cout << "Tensor shape: ";
    for (int i = 0; i < ndim; i++)
    {
        cout << shape[i] << " ";
    }
    cout << endl;

    gpu_print_(device, size, ptr);
    cout << endl;
}