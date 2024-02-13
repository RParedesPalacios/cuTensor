#include <iostream>
#include <vector>
#include "gpu.h"

using namespace std;


void msg(const char *s);

class cuTensor {

    private:
        unsigned int ndim;
        unsigned long int size;
        vector<int> shape;
        int device;
        float *ptr = nullptr;
    public:
        string name;

        cuTensor();
        cuTensor(const vector<int> &s, const int dev, const string n);
        cuTensor(const vector<int> &shape);
        cuTensor(const vector<int> &shape, const string n);
        cuTensor(const vector<int> &shape, const int dev);

        ~cuTensor();

    // Methods
    cuTensor *clone() const;
    void fill(float value);
    void info();
    void print();
    void reshape(const vector<int> &newshape);

    //OPS
    static cuTensor *sum(cuTensor *A, cuTensor *B);
    static cuTensor *mult2D(cuTensor *A, cuTensor *B);
};

typedef cuTensor* T;
typedef vector<int> tshape;

