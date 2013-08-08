#ifndef DEVICE_MATH_H
#define DEVICE_MATH_H

#include "types.h"

__device__ __host__
void print_matrix(double* A, int dims) ;

__device__ __host__
double safeLog( double x ) ;

__device__ __host__ double
logistic_function(double x, double lower, double upper, double beta, double tau) ;

__device__ void
matmultiply2(double *A, double *B, double *X) ;

__host__ __device__ double
det2(double *A) ;

__host__ __device__ double
det3(double *A) ;

__host__ __device__ double
det4(double *A) ;

__device__ __host__ void
invert_matrix2(double *A, double *A_inv) ;

__device__ void
invert_matrix3(double *A, double* A_inv) ;

__device__ void
invert_matrix4( double *A, double *Ainv) ;

__device__ __host__
void cholesky(double* A, double* L, int dims) ;

__device__ __host__ void
triangular_inverse(double *L, double *Linv, int dims) ;

__device__ double
evalGaussian(Gaussian2D g, double2 p) ;

__device__ double
evalLogGaussian(Gaussian2D g, double*p) ;

template<class GaussianType>
__device__ __host__ int
getGaussianDim(GaussianType g) ;

template<class GaussianType>
__device__ __host__ GaussianType
sumGaussians(GaussianType a, GaussianType b) ;

template<class GaussianType>
__device__ __host__ void
clearGaussian(GaussianType &a) ;

__host__ __device__ double
wrapAngle(double a) ;

__device__ void
makePositiveDefinite( double A[4] ) ;

template <int N, int N2>
__device__ double
computeMahalDist(Gaussian<N,N2> a, Gaussian<N,N2> b) ;

__device__ double
computeMahalDist(Gaussian2D a, Gaussian2D b) ;

__device__ double
computeMahalDist(Gaussian3D a, Gaussian3D b) ;

__device__ double
computeMahalDist(Gaussian4D a, Gaussian4D b) ;

template <class T>
__device__ double
computeHellingerDist(T a, T b) ;

__device__ double
computeHellingerDist( Gaussian2D a, Gaussian2D b) ;

__device__ void
sumByReduction( volatile double* sdata, double mySum, const unsigned int tid ) ;

__device__ void
productByReduction( volatile double* sdata, double my_factor, const unsigned int tid ) ;

__device__ void
maxByReduction( volatile double* sdata, double val, const unsigned int tid ) ;

__device__ double
logsumexpByReduction( volatile double* sdata, double val, const unsigned int tid ) ;

__device__ __host__
int sub_to_idx(int row, int col, int dim) ;

template<class GaussianType>
__device__ __host__
void copy_gaussians(GaussianType &src, GaussianType &dest) ;

template<class GaussianType>
__device__ __host__
void force_symmetric_covariance(GaussianType &g) ;

template<typename T>
__device__ __host__
T clampValue(T val, T abs_max, bool above = true) ;

template <int N, int N2>
__device__ __host__ bool
checkNan(Gaussian<N,N2> g) ;

template <int N, int N2>
__device__ __host__ bool
isPosDef(Gaussian<N,N2> g) ;

#endif // DEVICE_MATH_H
