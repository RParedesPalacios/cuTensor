#include <iostream>
#include <vector>
 
using namespace std;

class Tensor {
    private:
        unsigned int ndim;
        unsigned long int size;
        vector<int> shape;
        float *ptr = nullptr;
        bool is_shared = false;

    public:
        Tensor();
        Tensor(const vector<int> &s, float *ptr);
        Tensor(const vector<int> &shape);
    
        ~Tensor();

    /// Methods
    void fill(float value);
    void print();
    float *get_ptr() {return ptr;}
}; 