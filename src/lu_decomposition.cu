// LU matrix decomposition for square matrices with column-major storage
//
// Adapted from code found here: http://rosettacode.org/wiki/LU_decomposition#C
// Accessed 23 July 2013


#define _swap(x, y) { typeof(x) tmp = x; x = y; y = tmp; }

__device__ __host__
void mat_zero(double* x, int n) {
    for(int i = 0 ; i < n ; i++ ){
        for(int j = 0 ; j < n ; j++)
            x[i+j*n] = 0;
    }
}

__device__ __host__
void mat_eye(double* x, int n) {
    for(int i = 0 ; i < n ; i++ ){
        for(int j = 0 ; j < n ; j++)
            x[i+j*n] = (i==j);
    }
}

__device__ __host__
void mat_mul(double* a, double* b, double* c, int n)
{
    for(int i = 0 ; i < n ; i++ ){
        for(int j = 0 ; j < n ; j++){
            for(int k = 0 ; k < n ; k++){
                c[i+j*n] += a[i+k*n] * b[k+j*n ];
            }
        }
    }
}

__device__ __host__
void mat_pivot(double* a, double* p, int n)
{
    // create an identity matrix
    for ( int i = 0 ; i < n ; i++ ){
        for ( int j = 0 ; j < n ; j++){
            p[i+j*n] = (i == j) ;
        }
    }

    for ( int i = 0 ; i < n ; i++ ){
        int max_j = i ;
        for ( int j = i ; j < n ; j++ )
            if ((fabs(a[i+j*n]) - fabs(a[i+max_j*n])) > 0.00001) max_j = j;
        if ( max_j != i ){
            for(int k = 0 ; k < n ; k++){
                _swap(p[i+k*n], p[max_j+k*n]);
            }
        }
    }
}


__device__ __host__
void mat_LU(double* A, double* L, double* U, double* P, double* Aprime, int n)
{
    mat_zero(L,n);
    mat_zero(U,n);
    mat_pivot(A, P,n);


    mat_mul(P, A, Aprime,n);

    for ( int i = 0 ; i < n ; i++)  { L[i+i*n] = 1; }

    for ( int i = 0 ; i < n ; i++) {
        for ( int j = 0 ; j < n ; j++){
            double s;
            if (j <= i) {
                s = 0 ;
                for ( int k = 0 ; k < j ; k++){
                    s += L[j+k*n] * U[k+i*n] ;
                }
                U[j+i*n] = Aprime[j+i*n] - s;
            }
            if (j >= i) {
                s = 0 ;
                for ( int k = 0 ; k < i ; k++ ){
                    s += L[j+k*n] * U[k+i*n] ;
                }
                L[j+i*n] = (Aprime[j+i*n] - s) / U[i+i*n];
            }
        }
    }
}
