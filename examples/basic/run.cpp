#include <iostream>
#include "cutensor.h"
#include "tensor.h"
#include "gpu.h"

int main() {
    gpu_init();

    T t1,t2,t3,t4;
    
    t1=create({5,4,5},0);
    fill(t1,1.0);
    
    t2=create({5,3},0);
    fill(t2,1.0);

    t3=mult2D(t1,t2);

    info(t1);
    info(t2);
    info(t3);

    print(t3);

    delete t1;
    delete t2;
    delete t3;

    return 0;
}
