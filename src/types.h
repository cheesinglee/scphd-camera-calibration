#ifndef TYPES_H
#define TYPES_H

#include <cuda_runtime.h>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

typedef struct{
    float x ;
    float y ;
    float z ;
} Point3D ;

template <int N, int N2>
class Gaussian{
public:
    float weight ;
    float mean[N] ;
    float cov[N2] ;
    int dims ;

//    CUDA_CALLABLE_MEMBER
//    Gaussian(){
//        weight = 0 ;
//        dims = N ;
//        for ( int i = 0 ; i < N2 ; i++ ){
//            if ( i < N )
//                mean[i] = 0 ;
//            cov[i] = 0 ;
//        }
//    }

//    CUDA_CALLABLE_MEMBER
//    void
//    operator=(Gaussian other){
//        weight = other.weight ;
//        for (int i =0 ; i < N2 ; i++){
//            if ( i < N )
//                mean[i] = other.mean[i] ;
//            cov[i] = other.cov[i] ;
//        }
//    }
} ;

typedef Gaussian<2,4> Gaussian2D ;
typedef Gaussian<3,9> Gaussian3D ;
typedef Gaussian<4,16> Gaussian4D ;
typedef Gaussian<6,36> Gaussian6D ;

typedef struct{
    float x ;
    float y ;
    float z ;
    float vx ;
    float vy ;
    float vz ;
} EuclideanPoint ;

class DisparityPoint{
public:
    float u ;
    float v ;
    float d ;
    float vu ;
    float vv ;
    float vd ;

    CUDA_CALLABLE_MEMBER
    DisparityPoint operator-(DisparityPoint other){
        DisparityPoint result ;
        result.u = u - other.u ;
        result.v = v - other.v ;
        result.d = d - other.d ;
        result.vu = vu - other.vu ;
        result.vv = vv - other.vv ;
        result.vd = vd - other.vd ;
        return result ;
    }

    CUDA_CALLABLE_MEMBER
    DisparityPoint operator+(DisparityPoint other){
        DisparityPoint result ;
        result.u = u + other.u ;
        result.v = v + other.v ;
        result.d = d + other.d ;
        result.vu = vu + other.vu ;
        result.vv = vv + other.vv ;
        result.vd = vd + other.vd ;
        return result ;
    }

    template<typename T>
    CUDA_CALLABLE_MEMBER
    DisparityPoint operator*(const T a){
        DisparityPoint result ;
        result.u = a*u ;
        result.v = a*v ;
        result.d = a*d ;
        result.vu = a*vu ;
        result.vv = a*vv ;
        result.vd = a*vd ;
        return result ;
    }

    CUDA_CALLABLE_MEMBER
    DisparityPoint operator*(const DisparityPoint a){
        DisparityPoint result ;
        result.u = a.u*u ;
        result.v = a.v*v ;
        result.d = a.d*d ;
        result.vu = a.vu*vu ;
        result.vv = a.vv*vv ;
        result.vd = a.vd*vd ;
        return result ;
    }

    template<typename T>
    CUDA_CALLABLE_MEMBER
    DisparityPoint operator/(const T a){
        DisparityPoint result ;
        result.u = u/a ;
        result.v = v/a ;
        result.d = d/a ;
        result.vu = vu/a ;
        result.vv = vv/a ;
        result.vd = vd/a ;
        return result ;
    }

} ;

typedef struct{
    EuclideanPoint cartesian ;
    EuclideanPoint angular ;
} Extrinsics ;


#endif // TYPES_H
