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
    name = "";
    device = 0;
}
cuTensor::cuTensor(const vector<int> &s, const int dev, const string n)
{
    ndim = s.size();
    shape = s;
    device = dev;
    name = n;

    size=1;
    for (int i = ndim - 1; i >= 0; i--) {
        if (shape[i]<0) msg("error negative shape");
        size *= shape[i];
    }
    // initialize the stride
    strides.resize(ndim);
    strides[ndim-1]=1;
    for (int i = ndim - 2; i >= 0; i--) {
        strides[i]=strides[i+1]*shape[i+1];
    }

    ptr = gpu_create_tensor(device,size);
}

cuTensor::cuTensor(const vector<int> &shape):cuTensor(shape,0,""){}
cuTensor::cuTensor(const vector<int> &shape, const string n):cuTensor(shape,0,n){}
cuTensor::cuTensor(const vector<int> &shape, const int dev):cuTensor(shape,dev, ""){}

cuTensor::~cuTensor()
{
    gpu_delete_tensor(device,ptr);
}

////////////////////////////////////////
// Methods
////////////////////////////////////////
cuTensor *cuTensor::clone() const
{
    cuTensor *B = new cuTensor(shape, device);
    gpu_copy_(device, size, ptr, B->ptr);
    return B;
}

void cuTensor::fill(float value)
{
    gpu_fill(device, size, ptr, value);
}
void cuTensor::fill()
{
    gpu_fill_void(device, size, ptr);
}

void print_shape(tshape shape)
{
    cout << "Tensor shape: ";
    for (int i = 0; i < shape.size(); i++)
    {
        cout << shape[i] << " ";
    }
    cout << endl;
}

void cuTensor::info()
{
    cout << "--- Tensor Info ---\n";
    cout << "Tensor name: " << name << endl;
    cout << "Tensor in device: GPU " << device << endl;
    print_shape(shape);
}
void cuTensor::print()
{
    info();
    float *ptr2 = new float[size];
    gpu_copy_(device, size, ptr, ptr2);
    for (int i = 0; i < size; i++)
    {
        cout << fixed << setprecision(4) << ptr2[i] << " ";
    }   
    //gpu_print_(device, size, ptr);
    cout << endl;
    delete[] ptr2;
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
    strides.resize(ndim);
    strides[ndim-1]=1;
    for (int i = ndim - 2; i >= 0; i--) {
        strides[i]=strides[i+1]*shape[i+1];
    }
}


void cuTensor::permute(tshape perm)
{
    if (perm.size() != ndim) msg("error: permute must have the same number of dimensions\n");

    // check that in perm are all the dims
    for (int i = 0; i < ndim; i++)
    {
        bool found = false;
        for (int j = 0; j < ndim; j++)
        {
            if (perm[j] == i)
            {
                found = true;
                break;
            }
        }
        if (!found) msg("error: permute must contain all the dimensions\n");
    }

    // update shape
    tshape nshape(ndim);
    for (int i = 0; i < ndim; i++)
    {
        nshape[i] = shape[perm[i]];
    }
    shape = nshape;

    // new strides
    tshape nstrides(ndim);
    nstrides[ndim-1]=1;
    for (int i = ndim - 2; i >= 0; i--) {
        nstrides[i]=nstrides[i+1]*shape[i+1];
    }    
    gpu_permute_(device, size, ndim, strides.data(), nstrides.data(),perm.data(), ptr);

    strides=nstrides;
}

const std::vector<int>& cuTensor::getShape() const 
{
    return shape;
}
const std::vector<int>& cuTensor::getStride() const 
{
    return strides;
}
const int cuTensor::getDim() const 
{
    return ndim;
}
const int cuTensor::getDevice() const 
{
    return device;
}
const int cuTensor::getSize() const 
{
    return size;
}