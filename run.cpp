#include <iostream>
#include "cutensor.h"
#include "tensor.h"
#include "gpu.h"

int main() {
    T t1,t2,t3;
    gpu_init();
    
    t1=create({2,3,4},1);
    fill(t1,1.0);
    
    t2=create({6,4},1);
    fill(t2,2.0);

    t3=sum(t1,t2);
    print(t3);

    delete t1;
    delete t2;
    delete t3;
 
    return 0;
}
