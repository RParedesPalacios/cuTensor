#include <iostream>
#include <iomanip>
#include "cutensor.h"
#include "gpu.h"
#include "gpu_ops.h"

using namespace std;


void print_shape(const char* s, vector<int> shape)
{
    cout << s << " : ";
    for (int i = 0; i < shape.size(); i++)
    {
        cout << shape[i] << " ";
    }
    cout << endl;
}

///////////////////////////////////////////
// OPS
///////////////////////////////////////////
cuTensor * cuTensor::sum(cuTensor *A, cuTensor *B)
{
    if (A->size!=B->size) msg("error tensor size mismatch\n");
    if (A->device!=B->device) msg("error tensor device mismatch\n");

    cuTensor *C=new cuTensor(A->shape,A->device);
    gpu_sum(A->ptr,B->ptr,C->ptr,A->size,A->device,false);

    return C;
}

void find_match_dims(tshape a, tshape b, int &apos, int &bpos, int &match )
{
    int i,j;

    for (i=a.size()-1;i>=1;i--) {
        int m=1;
        for(j=i;j<a.size();j++) m*=a[j];
        
        int s=1;
        for(j=0;j<b.size()-1;j++) {
            s*=b[j];
            if (s==m) break;
        }

        if (j<b.size()-1) {
            match=m; apos=i; bpos=j;
            return;
        }
    }

    if (i<1) match=-1;
    return;
}

cuTensor * cuTensor::mult2D(cuTensor *A, cuTensor *B)
{
    
    tshape inis,ends;
    tshape As=A->shape;
    tshape Bs=B->shape;
    int apos,bpos,match;
    
    if (A->ndim<2) msg("error tensor size mismatch\n");
    if (B->ndim<2) msg("error tensor size mismatch\n");
    
    find_match_dims(A->shape, B->shape, apos,bpos,match);

    if (match==-1) msg("error tensor size mismatch\n");

    inis={A->shape.begin(),A->shape.begin()+apos};
    ends={B->shape.begin()+bpos+1,B->shape.end()};

    A->reshape({-1,match});
    B->reshape({match,-1});

    if (A->shape[1]!=B->shape[0]) msg("error tensor size mismatch\n");
    if (A->device!=B->device) msg("error tensor device mismatch\n");

    cuTensor *C=new cuTensor({A->shape[0],B->shape[1]},A->device);
    gpu_mult2D(A->ptr,B->ptr,C->ptr,A->shape[0],A->shape[1],B->shape[1],A->device);

    // create new shape concatenating inis and ends
    tshape newshape;
    newshape.insert(newshape.end(),inis.begin(),inis.end());
    newshape.insert(newshape.end(),ends.begin(),ends.end());

    C->reshape(newshape);
    A->reshape(As);
    B->reshape(Bs);

    return C;
}