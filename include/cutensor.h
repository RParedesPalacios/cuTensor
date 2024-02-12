#include <iostream>
#include <vector>
 
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
        cuTensor();
        cuTensor(const vector<int> &s, const int dev, float *cpu_ptr);
        cuTensor(const vector<int> &shape);
        cuTensor(const vector<int> &shape, float *cpu_ptr);
        cuTensor(const vector<int> &shape, const int dev);

        ~cuTensor();

    // Methods
    cuTensor *clone();
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

// wrapp OPS
cuTensor * create(const vector<int> &s, const int dev, float *cpu_ptr);
cuTensor * create(const vector<int> &s, const int dev);
cuTensor * create(const vector<int> &s, float *cpu_ptr);
cuTensor * create(const vector<int> &s);
cuTensor * clone(cuTensor *A);
void fill(cuTensor *A, float value);
void print(cuTensor *A);
void info(cuTensor *A);
void reshape(cuTensor *A, const vector<int> &newshape);

cuTensor * sum(cuTensor *A, cuTensor *B);
cuTensor * mult2D(cuTensor *A, cuTensor *B);
