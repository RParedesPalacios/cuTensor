#include <iostream>
#include <iomanip>
#include "cutensor.h"
#include "gpu_ops.h"

using namespace std;

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
void fill(cuTensor *A, float value) { A->fill(value);}
void info(cuTensor *A) { A->info();}
void print(cuTensor *A) { A->print();}
void reshape(cuTensor *A, const vector<int> &newshape) { A->reshape(newshape);}
cuTensor * clone(cuTensor *A) { return A->clone();}

cuTensor * sum(cuTensor *A, cuTensor *B) { return cuTensor::sum(A,B);}
cuTensor * mult2D(cuTensor *A, cuTensor *B) { return cuTensor::mult2D(A,B);}
