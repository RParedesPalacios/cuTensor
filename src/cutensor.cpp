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
cuTensor::cuTensor(const tshape &s, const int dev, const string n)
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

cuTensor::cuTensor(const tshape &shape):cuTensor(shape,0,""){}
cuTensor::cuTensor(const tshape &shape, const string n):cuTensor(shape,0,n){}
cuTensor::cuTensor(const tshape &shape, const int dev):cuTensor(shape,dev, ""){}

cuTensor::~cuTensor()
{
    gpu_delete_tensor(device,ptr);
}

////////////////////////////////////////
// Methods
////////////////////////////////////////
int cuTensor::get_ndim() {
    return ndim;
}

cuTensor *cuTensor::clone() const
{
    cuTensor *B = new cuTensor(shape, device);
    gpu_copy_to(device, size, ptr, B->ptr);
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

void cuTensor::info()
{
    cout << "--- Tensor Info ---\n";
    cout << "Tensor name: " << name << endl;
    cout << "Tensor in device: GPU " << device << endl;    
    
    cout << "Tensor shape: ";
    for (int i = 0; i < shape.size(); i++)
    {
        cout << shape[i] << " ";
    }
    cout << endl;
    cout << "Tensor size: " << size << endl;
}
void cuTensor::print()
{
    info();
    float *ptr2 = new float[size];
    gpu_copy_from(device, size, ptr, ptr2);
    for (int i = 0; i < size; i++)
    {
        cout << fixed << setprecision(4) << ptr2[i] << " ";
    }   
    //gpu_print_(device, size, ptr);
    cout << endl;
    delete[] ptr2;
}

string shape_to_string(const tshape &shape)
{
    string str = "";
    for (int i = 0; i < shape.size(); i++)
    {
        str += to_string(shape[i]) + " ";
    }
    return str;
}

string cuTensor::tostr()
{
    string str = ""; 
    str = str + "--- Tensor Info ---\n";
    str = str + "Tensor name: " + name + "\n";
    str = str + "Tensor in device: GPU " + to_string(device) + "\n";
    str = str + "Tensor shape: " + shape_to_string(shape) + "\n";
    str = str + "Tensor size: " + to_string(size) + "\n";
    return str;
}


const tshape& cuTensor::getShape() const 
{
    return shape;
}
const tshape& cuTensor::getStride() const 
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