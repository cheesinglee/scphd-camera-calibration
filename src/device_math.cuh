#ifndef DEVICE_MATH_H
#define DEVICE_MATH_H

#include "types.h"

__device__ __host__
float safeLog( float x ) ;

__device__ __host__ float
logistic_function(float x, float lower, float upper, float beta, float tau) ;

__device__ void
matmultiply2(float *A, float *B, float *X) ;

__host__ __device__ float
det2(float *A) ;

__host__ __device__ float
det3(float *A) ;

__host__ __device__ float
det4(float *A) ;

__device__ __host__ void
invert_matrix2(float *A, float *A_inv) ;

__device__ void
invert_matrix3(float *A, float* A_inv) ;

__device__ void
invert_matrix4( float *A, float *Ainv) ;

__device__ __host__
void cholesky(float* A, float* L, int dims) ;

__device__ __host__ void
triangular_inverse(float *L, float *Linv, int dims) ;

__device__ float
evalGaussian(Gaussian2D g, float2 p) ;

__device__ float
evalLogGaussian(Gaussian2D g, float*p) ;

template<class GaussianType>
__device__ __host__ int
getGaussianDim(GaussianType g) ;

template<class GaussianType>
__device__ __host__ GaussianType
sumGaussians(GaussianType a, GaussianType b) ;

template<class GaussianType>
__device__ __host__ void
clearGaussian(GaussianType &a) ;

__host__ __device__ float
wrapAngle(float a) ;

__device__ void
makePositiveDefinite( float A[4] ) ;

template <int N, int N2>
__device__ float
computeMahalDist(Gaussian<N,N2> a, Gaussian<N,N2> b) ;

__device__ float
computeMahalDist(Gaussian2D a, Gaussian2D b) ;

__device__ float
computeMahalDist(Gaussian3D a, Gaussian3D b) ;

__device__ float
computeMahalDist(Gaussian4D a, Gaussian4D b) ;

template <class T>
__device__ float
computeHellingerDist(T a, T b) ;

__device__ float
computeHellingerDist( Gaussian2D a, Gaussian2D b) ;

__device__ void
sumByReduction( volatile float* sdata, float mySum, const unsigned int tid ) ;

__device__ void
productByReduction( volatile float* sdata, float my_factor, const unsigned int tid ) ;

__device__ void
maxByReduction( volatile float* sdata, float val, const unsigned int tid ) ;

__device__ float
logsumexpByReduction( volatile float* sdata, float val, const unsigned int tid ) ;

__device__ __host__
int sub_to_idx(int row, int col, int dim) ;

template<class GaussianType>
__device__ __host__
void copy_gaussians(GaussianType &src, GaussianType &dest) ;

template<class GaussianType>
__device__ __host__
void force_symmetric_covariance(GaussianType &g) ;

#endif // DEVICE_MATH_H
