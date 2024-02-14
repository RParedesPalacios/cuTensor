#include <iostream>
#include <vector>
#include "gpu.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace std;

typedef vector<int> tshape;

void msg(const char *s);

class cuTensor {
    
    private:
        unsigned int ndim;
        unsigned long int size;
        vector<int> shape;
        int device;
        float *ptr = nullptr;

    // I want to add a new property to the class, the stride of the tensor
    vector<int> strides;

    public:
        string name;

        cuTensor();
        cuTensor(const vector<int> &s, const int dev, const string n);
        cuTensor(const vector<int> &shape);
        cuTensor(const vector<int> &shape, const string n);
        cuTensor(const vector<int> &shape, const int dev);

        ~cuTensor();

    // Methods
    int get_ndim();
    cuTensor *clone() const;
    void fill();
    void fill(float value);
    void info();
    void print();
    void reshape(const vector<int> &newshape);
    const std::vector<int>& getShape() const;
    const int getDim() const;
    const int getDevice() const;
    const int getSize() const;
    const std::vector<int>& getStride() const;
    void apply(py::function func, py::args args, py::kwargs kwargs);
    //OPS
    static cuTensor *sum(cuTensor *A, cuTensor *B);
    static cuTensor *mult2D(cuTensor *A, cuTensor *B);
    void contiguous(tshape nstride,tshape perm);
    void permute(tshape perm);

};



