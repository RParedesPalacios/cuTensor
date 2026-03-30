#include <iostream>
#include <iomanip>
#include "cutensor.h"
#include "gpu_ops.h"

using namespace std;

/// reshape tensor to a new shape, admiting "-1" e.g. (2,3,5) to (-1,5) should be (6,5) 
void cuTensor::reshape(const tshape &nshape)
{
    unsigned long int newsize = 1;
    int neg_index = -1;
    tshape newshape = nshape;

    for (int i = 0; i < newshape.size(); i++)
    {
        if (newshape[i] == -1)
        {
            if (neg_index != -1) msg("error: multiple occurrences of -1 in new shape\n");
            neg_index = i;
        }
        else {
            if (newshape[i] <= 0) msg("error: shape values must be positive or -1\n");
            newsize *= static_cast<unsigned long int>(newshape[i]);
        }
    }

    if (neg_index != -1)
    {
        if (newsize == 0 || size % newsize != 0) msg("error: cannot reshape tensor, size mismatch\n");
        int inferred = static_cast<int>(size / newsize);
        if (inferred <= 0) msg("error: inferred shape is not positive\n");
        newshape[neg_index] = inferred;
        newsize *= static_cast<unsigned long int>(inferred);
    }
    else
    {
        if (newsize != size) msg("error: cannot reshape tensor, size mismatch\n");
    }

    shape = newshape;
    ndim = shape.size();
    strides.clear();
    if (ndim > 0) {
        strides.resize(ndim);
        strides[ndim - 1] = 1;
        for (int i = static_cast<int>(ndim) - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
}



void cuTensor::squeeze()
{
    tshape nshape;
    for (int i = 0; i < ndim; i++)
    {
        if (shape[i] != 1) nshape.push_back(shape[i]);
    }
    if (nshape.size() == 0) nshape.push_back(1);
    reshape(nshape);
}

void cuTensor::unsqueeze(int axis)
{
    if (axis < 0 || axis > ndim) msg("error: axis out of range\n");
    tshape nshape;
    for (int i = 0; i < axis; i++) nshape.push_back(shape[i]);
    nshape.push_back(1);
    for (int i = axis; i < ndim; i++) nshape.push_back(shape[i]);
    reshape(nshape);
}



////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////
cuTensor* cuTensor::permute(tshape perm)
{
    cuTensor *C=clone();
    C->permute_(perm);
    return C;
}

void cuTensor::permute_(tshape perm)
{
    if (perm.size() != ndim) msg("error: permute must have the same number of dimensions\n");

    if (ndim == 0) return;

    // validate permutation indices [0, ndim)
    vector<bool> seen(ndim, false);
    for (int i = 0; i < static_cast<int>(ndim); i++)
    {
        int p = perm[i];
        if (p < 0 || p >= static_cast<int>(ndim)) msg("error: permute index out of range\n");
        if (seen[p]) msg("error: permute indices must be unique\n");
        seen[p] = true;
    }

    // update shape
    tshape nshape(ndim);
    for (int i = 0; i < static_cast<int>(ndim); i++)
    {
        nshape[i] = shape[perm[i]];
    }
    shape = nshape;

    // new strides
    tshape nstrides(ndim);
    nstrides[ndim - 1] = 1;
    for (int i = static_cast<int>(ndim) - 2; i >= 0; i--) {
        nstrides[i] = nstrides[i + 1] * shape[i + 1];
    }    
    gpu_permute_(device, size, ndim, strides.data(), nstrides.data(), perm.data(), ptr);

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

cuTensor * cuTensor::sumf(cuTensor *A, float s)
{
    cuTensor *C=new cuTensor(A->shape,A->device);
    gpu_sumf(A->ptr,C->ptr,A->size,s,A->device);
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

void cuTensor::mult2D_out(cuTensor *A, cuTensor *B, cuTensor *C)
{
    tshape inis, ends;
    tshape As = A->shape;
    tshape Bs = B->shape;
    int apos, bpos, match;

    if (A->ndim < 2) msg("error tensor size mismatch\n");
    if (B->ndim < 2) msg("error tensor size mismatch\n");
    if (A->device != B->device) msg("error tensor device mismatch\n");
    if (C->device != A->device) msg("error tensor device mismatch\n");

    find_match_dims(A->shape, B->shape, apos, bpos, match);
    if (match == -1) msg("error tensor size mismatch\n");

    inis = {A->shape.begin(), A->shape.begin() + apos};
    ends = {B->shape.begin() + bpos + 1, B->shape.end()};

    tshape outshape;
    outshape.insert(outshape.end(), inis.begin(), inis.end());
    outshape.insert(outshape.end(), ends.begin(), ends.end());

    if (C->shape != outshape) msg("error output tensor shape mismatch\n");

    A->reshape({-1, match});
    B->reshape({match, -1});

    if (A->shape[1] != B->shape[0]) msg("error tensor size mismatch\n");
    if (A->shape[0] * B->shape[1] != C->size) msg("error output tensor size mismatch\n");

    gpu_mult2D(A->ptr, B->ptr, C->ptr, A->shape[0], A->shape[1], B->shape[1], A->device);

    A->reshape(As);
    B->reshape(Bs);
}

cuTensor * cuTensor::mult(cuTensor *A, float s)
{
    cuTensor *C=new cuTensor(A->shape,A->device);
    gpu_mult(A->ptr,C->ptr,A->size,s,A->device);
    return C;
}

cuTensor * cuTensor::elementwise_product(cuTensor *A, cuTensor *B)
{
    if (A->size!=B->size) msg("error tensor size mismatch\n");
    if (A->device!=B->device) msg("error tensor device mismatch\n");

    cuTensor *C=new cuTensor(A->shape,A->device);
    gpu_elementwise_product(A->ptr,B->ptr,C->ptr,A->size,A->device);
    return C;
}

cuTensor * cuTensor::inv()
{
    cuTensor *C=new cuTensor(shape,device);
    gpu_inv(ptr,C->ptr,size,device);
    return C;
}

cuTensor * cuTensor::pow(float s)
{
    cuTensor *C=new cuTensor(shape,device);
    gpu_pow(ptr,C->ptr,size,s,device);
    return C;
}
