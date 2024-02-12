#include <iostream>
#include <iomanip>
#include "cutensor.h"

using namespace std;

void msg(const char *s)
{
    printf("%s",s);
    exit(1);
}
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
    for (int i = ndim - 1; i >= 0; i--) {
        if (shape[i]<0) msg("error negative shape");
        size *= shape[i];
    }

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
cuTensor *cuTensor::clone()
{
    cuTensor *B = new cuTensor(shape, device);
    gpu_copy_(device, size, ptr, B->ptr);
    return B;
}

void cuTensor::fill(float value)
{
    gpu_fill_(device, size, ptr, value);
}
void cuTensor::info()
{
    cout << "Tensor in device: GPU " << device << endl;
    cout << "Tensor shape: ";
    for (int i = 0; i < ndim; i++)
    {
        cout << shape[i] << " ";
    }
    cout << endl;
}
void cuTensor::print()
{
    info();
    gpu_print_(device, size, ptr);
    cout << endl;
}

/// reshape tensor to a new shape, admiting "-1" e.g. (2,3,5) to (-1,5) should be (6,5) 
void cuTensor::reshape(const vector<int> &nshape)
{
    int newsize = 1;
    int neg_index = -1;
    vector<int> newshape=nshape;

    for (int i = 0; i < newshape.size(); i++)
    {
        if (newshape[i] == -1)
        {
            if (neg_index != -1) msg("error: multiple occurrences of -1 in new shape\n");
            neg_index = i;
        }
        else newsize *= newshape[i];
    }

    if (neg_index != -1)
    {
        if (size % newsize != 0) msg("error: cannot reshape tensor, size mismatch\n");
        newshape[neg_index] = size / newsize;
    }
    else
    {
        if (newsize != size) msg("error: cannot reshape tensor, size mismatch\n");
    }

    shape = newshape;
    ndim = shape.size();
}
