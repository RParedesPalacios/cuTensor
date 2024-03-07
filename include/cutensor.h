#include <iostream>
#include <vector>
#include "gpu.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace std;

typedef vector<int> tshape;

void msg(const char *s);

class cuTensor {
    
    private:
        unsigned int ndim;
        unsigned long int size;
        tshape shape;
        int device;
        float *ptr = nullptr;

    // I want to add a new property to the class, the stride of the tensor
    tshape strides;

    public:
        string name;

        cuTensor();
        cuTensor(const tshape &s, const int dev, const string n);
        cuTensor(const tshape &shape);
        cuTensor(const tshape &shape, const string n);
        cuTensor(const tshape &shape, const int dev);
        cuTensor(const tshape &shape, float *ptr, const int dev, const string n);

        ~cuTensor();

    // Methods
    int get_ndim();
    cuTensor *clone() const;
    void fill();
    void fill(float value);
    void info();
    void print();
    string tostr();
    void reshape(const tshape &newshape);
    void squeeze();
    void unsqueeze(int axis);
    const tshape& getShape() const;
    const int getDim() const;
    const int getDevice() const;
    const int getSize() const;
    const tshape& getStride() const;    
    const string getName() const;
    void apply(py::function func, py::args args, py::kwargs kwargs);
    
    //OPS
    static cuTensor *sum(cuTensor *A, cuTensor *B);
    static cuTensor *sumf(cuTensor *A, float s);
    static cuTensor *mult2D(cuTensor *A, cuTensor *B);
    static cuTensor *mult(cuTensor *A, float s);
    static cuTensor *elementwise_product(cuTensor *A, cuTensor *B);
    
    cuTensor *inv();
    cuTensor *pow(float s);

    void contiguous(tshape nstride,tshape perm);
    void permute_(tshape perm);
    cuTensor* permute(tshape perm);
};



