#include <iostream>
#include <vector>
 
using namespace std;

class cuTensor {

    private:
        unsigned int ndim;
        unsigned long int size;
        vector<int> shape;
        int device;
        float *ptr = nullptr;
    public:
        cuTensor();
        cuTensor(const vector<int> &s, const int dev, float *cpu_ptr);
        cuTensor(const vector<int> &shape);
        cuTensor(const vector<int> &shape, float *cpu_ptr);
        cuTensor(const vector<int> &shape, const int dev);

        ~cuTensor();

    // Methods
    void fill(float value);
    void print();
};
