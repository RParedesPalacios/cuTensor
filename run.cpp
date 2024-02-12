#include <iostream>
#include "cutensor.h"
#include "tensor.h"
#include "gpu.h"

int main() {
    // Set the device to device 0
    gpu_init();
    
    Tensor *t1=new Tensor({2,3,4});
    t1->fill(1.0);
    t1->print();

    Tensor *t2=new Tensor({6,4},t1->get_ptr()); // sharing ptr
    t2->print();

    cuTensor *t=new cuTensor({2,3,4},t1->get_ptr());
    t->print();

    delete t;
 
    return 0;
}
