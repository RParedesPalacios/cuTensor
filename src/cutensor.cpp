#include <iostream>
#include <iomanip>
#include "cutensor.h"
#include "gpu.h"
#include "gpu_ops.h"

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

void msg(char *s)
{
    printf("%s",s);
    exit(1);
}

///////////////////////////////////////////
// OPS
///////////////////////////////////////////
cuTensor * cuTensor::sum(cuTensor *A, cuTensor *B)
{
    if (A->size!=B->size) msg("error tensor size mismatch\n");
    if (A->device!=B->device) msg("error tensor device mismatch\n");

    cuTensor *C=new cuTensor(A->shape,A->device);
    gpu_sum(A->ptr,B->ptr,C->ptr,A->size,A->device,false);

    return C;
}

///////////////////////////////////////////
// WRAPS
///////////////////////////////////////////
cuTensor * create(const vector<int> &s, const int dev, float *cpu_ptr)
{
    return new cuTensor(s,dev,cpu_ptr);
}
cuTensor * create(const vector<int> &s, const int dev)
{
    return new cuTensor(s,dev);
}
cuTensor * create(const vector<int> &s, float *cpu_ptr)
{
    return new cuTensor(s,cpu_ptr);
}
cuTensor * create(const vector<int> &s)
{
    return new cuTensor(s);
}

cuTensor * sum(cuTensor *A, cuTensor *B) { return cuTensor::sum(A,B);}

void fill(cuTensor *A, float value) { A->fill(value);}

void print(cuTensor *A) { A->print();}