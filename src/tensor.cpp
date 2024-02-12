#include <iostream>
#include <iomanip>
#include "tensor.h"


using namespace std;

////////////////////////////////////////
// Constructor and Destructor
////////////////////////////////////////
Tensor::Tensor()
{
    ndim = 0;
    size = 0;
    shape = {};
    ptr = nullptr;
}
Tensor::Tensor(const vector<int> &s, float *cpu_ptr)
{
    ndim = s.size();
    shape = s;

    size=1;
    for (int i = ndim - 1; i >= 0; i--) {size *= shape[i];}
    
    if (cpu_ptr != nullptr)
        ptr=cpu_ptr; // not copying, just using the pointer
    else {
        ptr = new float[size];
        is_shared = true;
    }
}
Tensor::Tensor(const vector<int> &shape):Tensor(shape,nullptr){}

Tensor::~Tensor()
{
    if (!is_shared) delete ptr;
}

////////////////////////////////////////
// Methods
////////////////////////////////////////
void Tensor::fill(float value)
{
    for (int i = 0; i < size; i++) ptr[i] = value;
}

void Tensor::print()
{
    cout << "Tensor in CPU: " <<  endl;
    cout << "Tensor size: " << size << endl;
    cout << "Tensor shape: ";

    for (int i = 0; i < ndim; i++) cout << shape[i] << " ";
    cout << endl;

    for (int i = 0; i < size; i++) printf("%f ",ptr[i]);
    printf("\n");
}