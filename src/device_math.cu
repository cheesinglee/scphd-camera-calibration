#include <float.h>
#include <stdio.h>
#include <curand_kernel.h>

#include "device_math.cuh"
#include "lu_decomposition.cuh"

#define LOG0 -FLT_MAX

__device__ __host__
void print_matrix(double* A, int dims){
    for ( int i = 0 ; i < dims ; i++){
        for ( int j = 0 ; j < dims ; j++){
            printf("%f ",A[i+j*dims]) ;
        }
        printf("\n") ;
    }
    printf("\n") ;
}

/// a nan-safe logarithm
__device__ __host__
double safeLog( double x )
{
    if ( x <= 0 )
        return LOG0 ;
    else
        return log(x) ;
}

/// evaluate generalized logistic function
__device__ __host__ double
logistic_function(double x, double lower, double upper, double beta, double tau)
{
    double y = (upper-lower)/(1+exp(-beta*(x-tau) ) ) ;
    return y ;
}

/// product of two 2x2 matrices
__device__ void
matmultiply2(double *A, double *B, double *X){
    X[0] = A[0]*B[0] + A[2]*B[1] ;
    X[1] = A[1]*B[0] + A[3]*B[1] ;
    X[2] = A[0]*B[2] + A[2]*B[3] ;
    X[3] = A[1]*B[2] + A[3]*B[3] ;
}

/// determinant of a 2x2 matrix
__host__ __device__ double
det2(double *A){
    return A[0]*A[3] - A[2]*A[1] ;
}

/// determinant of a 3x3 matrix
__host__ __device__ double
det3(double *A){
    return (A[0]*A[4]*A[8] + A[3]*A[7]*A[2] + A[6]*A[1]*A[5])
        - (A[0]*A[7]*A[5] + A[3]*A[1]*A[8] + A[6]*A[4]*A[2]) ;
}

/// determinant of a 4x4 matrix
__host__ __device__ double
det4(double *A)
{
    double det=0;
    det+=A[0]*((A[5]*A[10]*A[15]+A[9]*A[14]*A[7]+A[13]*A[6]*A[11])-(A[5]*A[14]*A[11]-A[9]*A[6]*A[15]-A[13]*A[10]*A[7]));
    det+=A[4]*((A[1]*A[14]*A[11]+A[9]*A[2]*A[15]+A[13]*A[10]*A[3])-(A[1]*A[10]*A[15]-A[9]*A[14]*A[3]-A[13]*A[2]*A[11]));
    det+=A[8]*((A[1]*A[6]*A[15]+A[5]*A[14]*A[3]+A[13]*A[2]*A[7])-(A[1]*A[14]*A[7]-A[5]*A[2]*A[15]-A[13]*A[6]*A[3]));
    det+=A[12]*((A[1]*A[10]*A[7]+A[5]*A[2]*A[12]+A[9]*A[10]*A[3])-(A[1]*A[10]*A[12]-A[5]*A[10]*A[3]-A[9]*A[2]*A[7]));
    return det ;
}

/// invert a 2x2 matrix
__device__ __host__ void
invert_matrix2(double *A, double *A_inv)
{
    double det = det2(A) ;
    A_inv[0] = A[3]/det ;
    A_inv[1] = -A[1]/det ;
    A_inv[2] = -A[2]/det ;
    A_inv[3] = A[0]/det ;
}

/// invert a 3x3 matrix
__device__ void
invert_matrix3(double *A, double* A_inv){
    double det = det3(A) ;
    A_inv[0] = (A[4]*A[8] - A[7]*A[5])/det ;
    A_inv[1] = (A[7]*A[2] - A[1]*A[8])/det ;
    A_inv[2] = (A[1]*A[5] - A[4]*A[2])/det ;
    A_inv[3] = (A[6]*A[5] - A[3]*A[8])/det ;
    A_inv[4] = (A[0]*A[8] - A[6]*A[2])/det ;
    A_inv[5] = (A[2]*A[3] - A[0]*A[5])/det ;
    A_inv[6] = (A[3]*A[7] - A[6]*A[4])/det ;
    A_inv[7] = (A[6]*A[1] - A[0]*A[7])/det ;
    A_inv[8] = (A[0]*A[4] - A[3]*A[1])/det ;
}

/// invert a 4x4 matrix
__device__ void
invert_matrix4( double *A, double *Ainv)
{
    Ainv[0] = (A[5] * A[15] * A[10] - A[5] * A[11] * A[14] - A[7] * A[13] * A[10] + A[11] * A[6] * A[13] - A[15] * A[6] * A[9] + A[7] * A[9] * A[14]) / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]);
    Ainv[1] = -(A[15] * A[10] * A[1] - A[11] * A[14] * A[1] + A[3] * A[9] * A[14] - A[15] * A[2] * A[9] - A[3] * A[13] * A[10] + A[11] * A[2] * A[13]) / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]);
    Ainv[2] = (A[5] * A[3] * A[14] - A[5] * A[15] * A[2] + A[15] * A[6] * A[1] + A[7] * A[13] * A[2] - A[3] * A[6] * A[13] - A[7] * A[1] * A[14]) / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]);
    Ainv[3] = -(A[5] * A[3] * A[10] - A[5] * A[11] * A[2] - A[3] * A[6] * A[9] - A[7] * A[1] * A[10] + A[11] * A[6] * A[1] + A[7] * A[9] * A[2]) / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]);
    Ainv[4] = -(A[15] * A[10] * A[4] - A[15] * A[6] * A[8] - A[7] * A[12] * A[10] - A[11] * A[14] * A[4] + A[11] * A[6] * A[12] + A[7] * A[8] * A[14]) / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]);
    Ainv[5] = (A[0] * A[15] * A[10] - A[0] * A[11] * A[14] + A[3] * A[8] * A[14] - A[15] * A[2] * A[8] + A[11] * A[2] * A[12] - A[3] * A[12] * A[10]) / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]);
    Ainv[6] = -(A[0] * A[15] * A[6] - A[0] * A[7] * A[14] - A[15] * A[2] * A[4] - A[3] * A[12] * A[6] + A[3] * A[4] * A[14] + A[7] * A[2] * A[12]) / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]);
    Ainv[7] = (-A[0] * A[7] * A[10] + A[0] * A[11] * A[6] + A[7] * A[2] * A[8] + A[3] * A[4] * A[10] - A[11] * A[2] * A[4] - A[3] * A[8] * A[6]) / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]);
    Ainv[8] = (-A[5] * A[15] * A[8] + A[5] * A[11] * A[12] + A[15] * A[4] * A[9] + A[7] * A[13] * A[8] - A[11] * A[4] * A[13] - A[7] * A[9] * A[12]) / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]);
    Ainv[9] = -(A[0] * A[15] * A[9] - A[0] * A[11] * A[13] - A[15] * A[1] * A[8] - A[3] * A[12] * A[9] + A[11] * A[1] * A[12] + A[3] * A[8] * A[13]) / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]);
    Ainv[10] = (A[15] * A[0] * A[5] - A[15] * A[1] * A[4] - A[3] * A[12] * A[5] - A[7] * A[0] * A[13] + A[7] * A[1] * A[12] + A[3] * A[4] * A[13]) / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]);
    Ainv[11] = -(A[11] * A[0] * A[5] - A[11] * A[1] * A[4] - A[3] * A[8] * A[5] - A[7] * A[0] * A[9] + A[7] * A[1] * A[8] + A[3] * A[4] * A[9]) / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]);
    Ainv[12] = -(-A[5] * A[8] * A[14] + A[5] * A[12] * A[10] - A[12] * A[6] * A[9] - A[4] * A[13] * A[10] + A[8] * A[6] * A[13] + A[4] * A[9] * A[14]) / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]);
    Ainv[13] = (-A[0] * A[13] * A[10] + A[0] * A[9] * A[14] + A[13] * A[2] * A[8] + A[1] * A[12] * A[10] - A[9] * A[2] * A[12] - A[1] * A[8] * A[14]) / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]);
    Ainv[14] = -(A[14] * A[0] * A[5] - A[14] * A[1] * A[4] - A[2] * A[12] * A[5] - A[6] * A[0] * A[13] + A[6] * A[1] * A[12] + A[2] * A[4] * A[13]) / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]);
    Ainv[15] = 0.1e1 / (A[0] * A[5] * A[15] * A[10] - A[0] * A[5] * A[11] * A[14] - A[0] * A[7] * A[13] * A[10] + A[0] * A[11] * A[6] * A[13] - A[0] * A[15] * A[6] * A[9] + A[0] * A[7] * A[9] * A[14] + A[5] * A[3] * A[8] * A[14] - A[5] * A[15] * A[2] * A[8] + A[5] * A[11] * A[2] * A[12] - A[5] * A[3] * A[12] * A[10] - A[15] * A[10] * A[1] * A[4] + A[15] * A[6] * A[1] * A[8] + A[15] * A[2] * A[4] * A[9] + A[3] * A[12] * A[6] * A[9] + A[7] * A[13] * A[2] * A[8] + A[7] * A[1] * A[12] * A[10] + A[3] * A[4] * A[13] * A[10] + A[11] * A[14] * A[1] * A[4] - A[11] * A[6] * A[1] * A[12] - A[11] * A[2] * A[4] * A[13] - A[3] * A[8] * A[6] * A[13] - A[7] * A[9] * A[2] * A[12] - A[7] * A[1] * A[8] * A[14] - A[3] * A[4] * A[9] * A[14]) * (A[10] * A[0] * A[5] - A[10] * A[1] * A[4] - A[2] * A[8] * A[5] - A[6] * A[0] * A[9] + A[6] * A[1] * A[8] + A[2] * A[4] * A[9]);
}

/// Lower Cholesky decomposition of a square matrix.
/// No check for positive-definiteness is performed.
__device__ __host__
void cholesky(double* A, double* L, int dims){
    for ( int i = 0; i < dims*dims ; i++ )
        L[i] = 0.0 ;

    L[0] = sqrt(A[0]) ;
    for (int i = 0 ; i < dims ; i++){
        for ( int j = 0 ; j <= i ; j++){
            int ij = i + j*dims ;
            double tmp = A[ij] ;
            if ( i == j ){
                for (int k = 0 ; k < j ; k++){
                    int jk = j + k*dims ;
                    tmp -= L[jk]*L[jk] ;
                }
                L[ij] = sqrt(tmp) ;
            }
            else{
                for ( int k = 0 ; k < j ; k++){
                    int ik = i + k*dims ;
                    int jk = j + k*dims ;
                    tmp -= L[ik]*L[jk] ;
                }
                int jj = j + j*dims ;
                L[ij] = tmp/L[jj] ;
            }
        }
    }
}

__device__ __host__ void
triangular_inverse(double *L, double *Linv, int dims){
    // solve for the columns of the inverse using forward substitution
    for (int col = 0 ; col < dims ; col++ ){
        for ( int i = 0 ; i < dims ; i++ ){
            if ( i >= col ){
                double val ;
                if ( i == col )
                    val = 1 ;
                else
                    val = 0 ;

                for( int j = 0 ; j < i ; j++ )
                    val -= L[i + j*dims]*Linv[j+col*dims] ;

                Linv[i+col*dims] = val/L[i+i*dims] ;
            }
            else{
                Linv[i+col*dims] = 0.0 ;
            }
        }
    }
}

__device__ __host__ void
triangular_inverse_upper(double *U, double *Uinv, int dims){
    // solve for the columns of the inverse using backward substitution
    for (int col = dims-1 ; col >= 0  ; col-- ){
        for ( int i = dims-1 ; i >= 0  ; i-- ){
            if ( i <= col ){
                double val ;
                if ( i == col )
                    val = 1 ;
                else
                    val = 0 ;

                for( int j = dims-1 ; j > i ; j-- )
                    val -= U[i + j*dims]*Uinv[j+col*dims] ;

                Uinv[i+col*dims] = val/U[i+i*dims] ;
            }
            else{
                Uinv[i+col*dims] = 0.0 ;
            }
        }
    }
}

/// evaluate the product x*A*x'
__device__ __host__ double
quadratic_matrix_product(double* A, double *x, int dims){
    double result = 0 ;
    for ( int i = 0 ; i < dims ; i++){
        double val = 0 ;
        for ( int j = 0 ; j < dims ; j++ ){
            val += x[j]*A[i+j*dims] ;
        }
        result += x[i]*val ;
    }
    return result ;
}

__device__ __host__ void
fill_identity_matrix(double* A, int dims){
    for ( int i = 0; i < dims ;i++){
        for(int j =0; j < dims ;j++){
            A[i+j*dims] = (i==j) ;
        }
    }
}

__device__ double
evalGaussian(Gaussian2D g, double2 p){
    // distance from mean
    double d[2] ;
    d[0] = g.mean[0] - p.x ;
    d[1] = g.mean[1] - p.y ;

    // inverse covariance matrix
    double S_inv[4] ;
    invert_matrix2(g.cov,S_inv);

    // determinant of covariance matrix
    double det_S = det2(g.cov) ;

    // compute exponential
    double exponent = 0.5*(d[0]*d[0]*S_inv[0]
            + d[0]*d[1]*(S_inv[1]+S_inv[2])
            + d[1]*d[1]*S_inv[3]) ;

    return exp(exponent)/sqrt(det_S)/(2*M_PI)*g.weight ;
}

__device__ double
evalLogGaussian(Gaussian2D g, double*p){
    // distance from mean
    double d[2] ;
    d[0] = g.mean[0] - p[0] ;
    d[1] = g.mean[1] - p[1] ;

    // inverse covariance matrix
    double S_inv[4] ;
    invert_matrix2(g.cov,S_inv);

    // determinant of covariance matrix
    double det_S = det2(g.cov) ;

    // compute exponential
    double exponent = 0.5*(d[0]*d[0]*S_inv[0]
            + d[0]*d[1]*(S_inv[1]+S_inv[2])
            + d[1]*d[1]*S_inv[3]) ;

    return exponent - safeLog(sqrt(det_S)) - safeLog(2*M_PI) +
                safeLog(g.weight) ;
}

template<class GaussianType>
__device__ __host__ int
getGaussianDim(GaussianType g)
{
    return int(sizeof(g.mean)/sizeof(double)) ;
//    return g.dims ;
}

template<class GaussianType>
__device__ __host__ GaussianType
sumGaussians(GaussianType a, GaussianType b)
{
    GaussianType result ;
    int dims = getGaussianDim(a) ;
    for (int i = 0 ; i < dims*dims ; i++ )
    {
        if (i < dims)
            result.mean[i] = a.mean[i] + b.mean[i] ;
        result.cov[i] = a.cov[i] + b.cov[i] ;
    }
    result.weight = a.weight + b.weight ;
    return result ;
}

template<class GaussianType>
__device__ __host__ void
clearGaussian(GaussianType &a)
{
    int dims = getGaussianDim(a) ;
    a.weight = 0 ;
    for (int i = 0 ; i < dims*dims ; i++)
    {
        if (i < dims)
            a.mean[i] = 0 ;
        a.cov[i] = 0 ;
    }
}

/// wrap an angular value to the range [-pi,pi]
__host__ __device__ double
wrapAngle(double a)
{
    double remainder = fmod(a, double(2*M_PI)) ;
    if ( remainder > M_PI )
        remainder -= 2*M_PI ;
    else if ( remainder < -M_PI )
        remainder += 2*M_PI ;
    return remainder ;
}

/// return the closest symmetric positve definite matrix for 2x2 input
__device__ void
makePositiveDefinite( double A[4] )
{
    // eigenvalues:
    double detA = A[0]*A[3] + A[1]*A[2] ;
    // check if already positive definite
    if ( detA > 0 && A[0] > 0 )
    {
        A[1] = (A[1] + A[2])/2 ;
        A[2] = A[1] ;
        return ;
    }
    double trA = A[0] + A[3] ;
    double trA2 = trA*trA ;
    double eval1 = 0.5*trA + 0.5*sqrt( trA2 - 4*detA ) ;
    double eval2 = 0.5*trA - 0.5*sqrt( trA2 - 4*detA ) ;

    // eigenvectors:
    double Q[4] ;
    if ( fabs(A[1]) > 0 )
    {
        Q[0] = eval1 - A[3] ;
        Q[1] = A[1] ;
        Q[2] = eval2 - A[3] ;
        Q[3] = A[1] ;
    }
    else if ( fabs(A[2]) > 0 )
    {
        Q[0] = A[2] ;
        Q[1] = eval1 - A[0] ;
        Q[2] = A[2] ;
        Q[3] = eval2 - A[0] ;
    }
    else
    {
        Q[0] = 1 ;
        Q[1] = 0 ;
        Q[2] = 0 ;
        Q[3] = 1 ;
    }

    // make eigenvalues positive
    if ( eval1 < 0 )
        eval1 = DBL_EPSILON ;
    if ( eval2 < 0 )
        eval2 = DBL_EPSILON ;

    // compute the approximate matrix
    A[0] = Q[0]*Q[0]*eval1 + Q[2]*Q[2]*eval2 ;
    A[1] = Q[0]*eval1*Q[1] + Q[2]*eval2*Q[3] ;
    A[2] = A[1] ;
    A[3] = Q[1]*Q[1]*eval1 + Q[3]*Q[3]*eval2 ;
}

template <int N, int N2>
__device__ double
computeMahalDist(Gaussian<N,N2> a, Gaussian<N,N2> b){
    // innovation vector
    double innov[N] ;
    for ( int i = 0 ; i < N ; i++ )
        innov[i] = a.mean[i] - b.mean[i] ;

    // innovation covariance
    double sigma[N2] ;
//    double trace_a = 0 ;
//    double trace_b = 0 ;

//    for ( int i = 0 ; i < N ; i++){
//        trace_a += a.cov[i+i*N] ;
//        trace_b += b.cov[i+i*N] ;
//    }

//    if (trace_a < trace_b){
//        for (int i = 0 ; i < N2 ; i++)
//            sigma[i] = a.cov[i] ;
//    }
//    else{
//        for (int i = 0 ; i < N2 ; i++)
//            sigma[i] = b.cov[i] ;
//    }

    for (int i = 0 ; i < N2 ; i++)
        sigma[i] = a.cov[i] + b.cov[i] ;

//    // LU decomposition and inverse
//    double L[N2] ;
//    double U[N2] ;
//    double P[N2] ;
//    double AA[N2] ;
//    mat_LU(sigma,L,U,P,AA,N) ;

//    double Linv[N2] ;
//    double Uinv[N2] ;
//    triangular_inverse(L,Linv,N) ;
//    triangular_inverse_upper(U,Uinv,N) ;

//    // re-use matrices L and U for intermediate product storage
//    double tmp[N2] ; double tmp2[N2] ;
//    mat_mul(Uinv,Linv,tmp,N);
//    mat_mul(tmp,P,tmp2,N) ;

////    if (threadIdx.x == 4){
////        print_matrix(sigma,N);
////        print_matrix(L,N);
////        print_matrix(U,N);
////        print_matrix(P,N);
////        print_matrix(Linv,N);
////        print_matrix(Uinv,N);
////        print_matrix(tmp,N);
////        print_matrix(tmp2,N);

////    }

//    return quadratic_matrix_product(tmp2,innov,N) ;

    double L[N2] ;
    cholesky(sigma,L,N);

    double Linv[N2] ;
    triangular_inverse(L,Linv,N) ;


    // multiply innovation with inverse L
    // distance is sum of squares
    double dist = 0 ;
    for ( int i = 0 ; i < N ; i++ ){
        double sum = 0 ;
        for ( int j = 0 ; j <= i ; j++){
            sum += innov[j]*Linv[i+j*N] ;
        }
        dist += sum*sum ;
    }
    return dist ;
}

/// compute the Mahalanobis distance between two Gaussians
__device__ double
computeMahalDist(Gaussian2D a, Gaussian2D b)
{
    double dist = 0 ;
    double sigma_inv[4] ;
    double sigma[4] ;
    for (int i = 0 ; i <4 ; i++)
        sigma[i] = (a.cov[i] + b.cov[i])/2 ;
    invert_matrix2(sigma,sigma_inv);
    double innov[2] ;
    innov[0] = a.mean[0] - b.mean[0] ;
    innov[1] = a.mean[1] - b.mean[1] ;
    dist = innov[0]*innov[0]*sigma_inv[0] +
            innov[0]*innov[1]*(sigma_inv[1]+sigma_inv[2]) +
            innov[1]*innov[1]*sigma_inv[3] ;
    return dist ;
}

__device__ double
computeMahalDist(Gaussian3D a, Gaussian3D b)
{
    double dist = 0 ;
    double sigma_inv[9] ;
    double sigma[9] ;
    for (int i = 0 ; i <9 ; i++)
        sigma[i] = (a.cov[i] + b.cov[i])/2 ;
    invert_matrix3(sigma,sigma_inv);
    double innov[3] ;
    innov[0] = a.mean[0] - b.mean[0] ;
    innov[1] = a.mean[1] - b.mean[1] ;
    innov[2] = a.mean[1] - b.mean[1] ;
    dist = innov[0]*(sigma_inv[0]*innov[0] + sigma_inv[3]*innov[1] + sigma_inv[6]*innov[2])
            + innov[1]*(sigma_inv[1]*innov[0] + sigma_inv[4]*innov[1] + sigma_inv[7]*innov[2])
            + innov[2]*(sigma_inv[2]*innov[0] + sigma_inv[5]*innov[1] + sigma_inv[8]*innov[2]) ;
    return dist ;
}

__device__ double
computeMahalDist(Gaussian4D a, Gaussian4D b)
{
    double dist = 0 ;
    double sigma_inv[16] ;
    double sigma[16] ;
    for (int i = 0 ; i < 16 ; i++)
        sigma[i] = (a.cov[i] + b.cov[i])/2 ;
    invert_matrix4(sigma,sigma_inv) ;
    double innov[4] ;
    for ( int i = 0 ; i < 4 ; i++ )
        innov[i] = a.mean[i] - b.mean[i] ;
    dist = innov[0]*(sigma_inv[0]*innov[0] + sigma_inv[4]*innov[1] + sigma_inv[8]*innov[2] + sigma_inv[12]*innov[3])
            + innov[1]*(sigma_inv[1]*innov[0] + sigma_inv[5]*innov[1] + sigma_inv[9]*innov[2] + sigma_inv[13]*innov[3])
            + innov[2]*(sigma_inv[2]*innov[0] + sigma_inv[6]*innov[1] + sigma_inv[10]*innov[2] + sigma_inv[14]*innov[3])
            + innov[3]*(sigma_inv[3]*innov[0] + sigma_inv[7]*innov[1] + sigma_inv[11]*innov[2] + sigma_inv[15]*innov[3]) ;
    return dist ;
}

/// Compute the Hellinger distance between two Gaussians
template <class T>
__device__ double
computeHellingerDist(T a, T b)
{
    return 0 ;
}

__device__ double
computeHellingerDist( Gaussian2D a, Gaussian2D b)
{
    double dist = 0 ;
    double innov[2] ;
    double sigma[4] ;
    double detSigma ;
    double sigmaInv[4] = {1,0,0,1} ;
    innov[0] = a.mean[0] - b.mean[0] ;
    innov[1] = a.mean[1] - b.mean[1] ;
    sigma[0] = a.cov[0] + b.cov[0] ;
    sigma[1] = a.cov[1] + b.cov[1] ;
    sigma[2] = a.cov[2] + b.cov[2] ;
    sigma[3] = a.cov[3] + b.cov[3] ;
    detSigma = det2(sigma) ;
    if (detSigma > FLT_MIN)
    {
        sigmaInv[0] = sigma[3]/detSigma ;
        sigmaInv[1] = -sigma[1]/detSigma ;
        sigmaInv[2] = -sigma[2]/detSigma ;
        sigmaInv[3] = sigma[0]/detSigma ;
    }
    double epsilon = -0.25*
            (innov[0]*innov[0]*sigmaInv[0] +
             innov[0]*innov[1]*(sigmaInv[1]+sigmaInv[2]) +
             innov[1]*innov[1]*sigmaInv[3]) ;

    // determinant of half the sum of covariances
    detSigma /= 4 ;
    dist = 1/detSigma ;

    // product of covariances
    sigma[0] = a.cov[0]*b.cov[0] + a.cov[2]*b.cov[1] ;
    sigma[1] = a.cov[1]*b.cov[0] + a.cov[3]*b.cov[1] ;
    sigma[2] = a.cov[0]*b.cov[2] + a.cov[2]*b.cov[3] ;
    sigma[3] = a.cov[1]*b.cov[2] + a.cov[3]*b.cov[3] ;
    detSigma = det2(sigma) ;
    dist *= sqrt(detSigma) ;
    dist = 1 - sqrt(dist)*exp(epsilon) ;
    return dist ;
}


//__device__ void
//cholesky( double*A, double* L, int size)
//{
//    int i = size ;
//    int n_elements = 0 ;
//    while(i > 0)
//    {
//        n_elements += i ;
//        i-- ;
//    }

//    int diag_idx = 0 ;
//    int diag_inc = size ;
//    L[0] = sqrt(A[0]) ;
//    for ( i = 0 ; i < n_elements ; i++ )
//    {
//        if (i==diag_idx)
//        {
//            L[i] = A[i] ;
//            diag_idx += diag_inc ;
//            diag_inc-- ;
//        }
//    }
//}


/// device function for summations by parallel reduction in shared memory
/*!
  * Implementation based on NVIDIA whitepaper found at:
  * http://developer.download.nvidia.com/compute/cuda/1_1/Website/projects/reduction/doc/reduction.pdf
  *
  * Result is stored in sdata[0]
  \param sdata pointer to shared memory array
  \param mySum summand loaded by the thread
  \param tid thread index
  */
__device__ void
sumByReduction( volatile double* sdata, double mySum, const unsigned int tid )
{
    sdata[tid] = mySum;
    __syncthreads();

    // do reduction in shared mem
    if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } __syncthreads();
    if (tid <  64) { sdata[tid] = mySum = mySum + sdata[tid +  64]; } __syncthreads();

    if (tid < 32)
    {
        sdata[tid] = mySum = mySum + sdata[tid + 32];
        sdata[tid] = mySum = mySum + sdata[tid + 16];
        sdata[tid] = mySum = mySum + sdata[tid +  8];
        sdata[tid] = mySum = mySum + sdata[tid +  4];
        sdata[tid] = mySum = mySum + sdata[tid +  2];
        sdata[tid] = mySum = mySum + sdata[tid +  1];
    }
    __syncthreads() ;
}

/// device function for products by parallel reduction in shared memory
/*!
  * Implementation based on NVIDIA whitepaper found at:
  * http://developer.download.nvidia.com/compute/cuda/1_1/Website/projects/reduction/doc/reduction.pdf
  *
  * Result is stored in sdata[0]
  \param sdata pointer to shared memory array
  \param my_factor factor loaded by the thread
  \param tid thread index
  */
__device__ void
productByReduction( volatile double* sdata, double my_factor, const unsigned int tid )
{
    sdata[tid] = my_factor;
    __syncthreads();

    // do reduction in shared mem
    if (tid < 128) { sdata[tid] = my_factor = my_factor * sdata[tid + 128]; } __syncthreads();
    if (tid <  64) { sdata[tid] = my_factor = my_factor * sdata[tid +  64]; } __syncthreads();

    if (tid < 32)
    {
        sdata[tid] = my_factor = my_factor * sdata[tid + 32];
        sdata[tid] = my_factor = my_factor * sdata[tid + 16];
        sdata[tid] = my_factor = my_factor * sdata[tid +  8];
        sdata[tid] = my_factor = my_factor * sdata[tid +  4];
        sdata[tid] = my_factor = my_factor * sdata[tid +  2];
        sdata[tid] = my_factor = my_factor * sdata[tid +  1];
    }
    __syncthreads() ;
}

/// device function for finding max value by parallel reduction in shared memory
/*!
  * Implementation based on NVIDIA whitepaper found at:
  * http://developer.download.nvidia.com/compute/cuda/1_1/Website/projects/reduction/doc/reduction.pdf
  *
  * Result is stored in sdata[0]. Other values in the array are garbage.
  \param sdata pointer to shared memory array
  \param val value loaded by the thread
  \param tid thread index
  */
__device__ void
maxByReduction( volatile double* sdata, double val, const unsigned int tid )
{
    sdata[tid] = val ;
    __syncthreads();

    // do reduction in shared mem
    if (tid < 128) { sdata[tid] = val = fmax(sdata[tid+128],val) ; } __syncthreads();
    if (tid <  64) { sdata[tid] = val = fmax(sdata[tid+64],val) ; } __syncthreads();

    if (tid < 32)
    {
        sdata[tid] = val = fmax(sdata[tid+32],val) ;
        sdata[tid] = val = fmax(sdata[tid+16],val) ;
        sdata[tid] = val = fmax(sdata[tid+8],val) ;
        sdata[tid] = val = fmax(sdata[tid+4],val) ;
        sdata[tid] = val = fmax(sdata[tid+2],val) ;
        sdata[tid] = val = fmax(sdata[tid+1],val) ;
    }
    __syncthreads() ;
}

__device__ double
logsumexpByReduction( volatile double* sdata, double val, const unsigned int tid )
{
    maxByReduction( sdata, val, tid ) ;
    double maxval = sdata[0] ;
    __syncthreads() ;

    sumByReduction( sdata, exp(val-maxval), tid) ;
    return safeLog(sdata[0]) + maxval ;
}


__device__ __host__
int sub_to_idx(int row, int col, int dim)
{
    int idx = row + col*dim ;
    return idx ;
}

template<class GaussianType>
__device__ __host__
void copy_gaussians(GaussianType &src, GaussianType &dest)
{
    // determine the size of the covariance matrix
    int dims = getGaussianDim(src) ;
    // copy mean and covariance
    for (int i = 0 ; i < dims*dims ; i++ )
    {
        if ( i < dims )
            dest.mean[i] = src.mean[i] ;
        dest.cov[i] = src.cov[i] ;
    }

    // copy weight
    dest.weight = src.weight ;
}

template<class GaussianType>
__device__ __host__
void force_symmetric_covariance(GaussianType &g)
{
    int dims = getGaussianDim(g) ;
    for ( int i = 0 ; i < dims ; i++ )
    {
        for( int j = 0 ; j < i ; j++)
        {
            int idx_lower = sub_to_idx(i,j,dims) ;
            int idx_upper = sub_to_idx(j,i,dims) ;
            g.cov[idx_lower] = (g.cov[idx_lower] + g.cov[idx_upper])/2 ;
            g.cov[idx_upper] = g.cov[idx_lower] ;
        }
    }
}

template<typename T>
__device__ __host__
T clampValue(T val, T clamp_val, bool above){
    T clamped = 0 ;

    if (clamp_val < 0){
        clamp_val = -clamp_val ;
    }

    bool do_clamp = false ;
    if (above)
        do_clamp = (fabs(val) > clamp_val) ;
    else
        do_clamp = (fabs(val) < clamp_val) ;

    if (do_clamp)
    {
        if (val >= 0)
            clamped = clamp_val ;
        else
            clamped = -clamp_val ;
    }
    return clamped ;
}

template <int N, int N2>
__device__ __host__ bool
checkNan(Gaussian<N,N2> g){
    bool ret = false ;
    if ( isnan(g.weight) ){
        ret = true  ;
    }
    else{
        for (int i = 0 ; i < N2 ; i++){
            if ( i < N && isnan(g.mean[i]) ){
                ret = true  ;
                break ;
            }
            else if ( isnan(g.cov[i]) ){
                ret = true ;
                break ;
            }
        }
    }

    if (ret){
        printf("nan feature detected\n") ;
        printf("w = %f, m =",g.weight) ;
        for ( int n = 0 ; n < N ; n++)
            printf(" %f",g.mean[n]) ;
        printf("\n") ;
        print_matrix(g.cov,N);
    }

    return ret ;
}


/// Test for positive definiteness.
/// Attempts a cholesky decomposition on covariance matrix
/// and checks that resulting decomposition is valid
template <int N, int N2>
__device__ __host__ bool
isPosDef(Gaussian<N,N2> g){
    double L[N2] ;
    cholesky(g.cov, L, N);

    for ( int i = 0 ; i < N2 ; i++ )
    {
        if ( isnan(L[i]) )
            return false ;
    }
    return true ;
}

// explicit template instantiation
template __device__ __host__
void force_symmetric_covariance(Gaussian6D &g) ;

template __device__ __host__ void
clearGaussian(Gaussian6D &g) ;

template __device__ double
computeMahalDist(Gaussian6D a, Gaussian6D b) ;

template __device__ __host__ void
copy_gaussians(Gaussian6D &src, Gaussian6D &dest) ;

template __device__ __host__ double
clampValue(double val, double abs_max, bool above) ;

template __device__ __host__ bool
checkNan(Gaussian6D g) ;

template __device__ __host__ bool
isPosDef(Gaussian6D g) ;

