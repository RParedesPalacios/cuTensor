#include <iostream>
#include <iomanip>
#include "cutensor.h"
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

/// reshape tensor to a new shape, admiting "-1" e.g. (2,3,5) to (-1,5) should be (6,5) 
void cuTensor::reshape(const vector<int> &nshape)
{
    int newsize = 1;
    int neg_index = -1;
    vector<int> newshape=nshape;

    for (int i = 0; i < newshape.size(); i++)
    {
        if (newshape[i] == -1)
        {
            if (neg_index != -1) msg("error: multiple occurrences of -1 in new shape\n");
            neg_index = i;
        }
        else newsize *= newshape[i];
    }

    if (neg_index != -1)
    {
        if (size % newsize != 0) msg("error: cannot reshape tensor, size mismatch\n");
        newshape[neg_index] = size / newsize;
    }
    else
    {
        if (newsize != size) msg("error: cannot reshape tensor, size mismatch\n");
    }

    shape = newshape;
    ndim = shape.size();
    strides.resize(ndim);
    strides[ndim-1]=1;
    for (int i = ndim - 2; i >= 0; i--) {
        strides[i]=strides[i+1]*shape[i+1];
    }
}

////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////

void cuTensor::permute(tshape perm)
{
    if (perm.size() != ndim) msg("error: permute must have the same number of dimensions\n");

    // check that in perm are all the dims
    for (int i = 0; i < ndim; i++)
    {
        bool found = false;
        for (int j = 0; j < ndim; j++)
        {
            if (perm[j] == i)
            {
                found = true;
                break;
            }
        }
        if (!found) msg("error: permute must contain all the dimensions\n");
    }

    // update shape
    tshape nshape(ndim);
    for (int i = 0; i < ndim; i++)
    {
        nshape[i] = shape[perm[i]];
    }
    shape = nshape;

    // new strides
    tshape nstrides(ndim);
    nstrides[ndim-1]=1;
    for (int i = ndim - 2; i >= 0; i--) {
        nstrides[i]=nstrides[i+1]*shape[i+1];
    }    
    gpu_permute_(device, size, ndim, strides.data(), nstrides.data(),perm.data(), ptr);

    strides=nstrides;
}


///////////////////////////////////////////
// OPS
///////////////////////////////////////////
cuTensor * cuTensor::sum(cuTensor *A, cuTensor *B)
{
    long int As=A->size;
    long int Bs=B->size;
    cuTensor *C;
    
    if (A->device!=B->device) msg("error tensor device mismatch\n");
    
    if (As>Bs) {
        if (As%Bs!=0) msg("error tensor size mismatch\n");
        C=new cuTensor(A->shape,A->device);
        gpu_sum(A->ptr,B->ptr,C->ptr,As,Bs,A->device,false);
    }
    else {
        if (Bs%As!=0) msg("error tensor size mismatch\n");
        C=new cuTensor(B->shape,B->device);
        gpu_sum(B->ptr,A->ptr,C->ptr,Bs,As,B->device,false);
    }

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