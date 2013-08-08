#ifndef LU_DECOMPOSITION_CUH
#define LU_DECOMPOSITION_CUH

__device__ __host__
void mat_zero(double* x, int n) ;

__device__ __host__
void mat_eye(double* x, int n) ;

__device__ __host__
void mat_mul(double* a, double* b, double* c, int n) ;

__device__ __host__
void mat_pivot(double* a, double* p, int n) ;

__device__ __host__
void mat_LU(double* A, double* L, double* U, double* P, double* Aprime, int n) ;

#endif // LU_DECOMPOSITION_CUH
