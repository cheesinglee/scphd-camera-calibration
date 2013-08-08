#ifndef DISPARITYMEASUREMENTMODEL_CUH
#define DISPARITYMEASUREMENTMODEL_CUH

#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include "types.h"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

class DisparityMeasurementModel
{
public:
    CUDA_CALLABLE_MEMBER DisparityMeasurementModel();

    CUDA_CALLABLE_MEMBER DisparityMeasurementModel(double fx, double fy, double u0, double v0,
                             double std_u, double std_v, double pd, double lambda);

    CUDA_CALLABLE_MEMBER DisparityPoint
    computeMeasurement(EuclideanPoint p_world);

    CUDA_CALLABLE_MEMBER DisparityPoint
    computeMeasurement(EuclideanPoint p_world, Extrinsics e);

    CUDA_CALLABLE_MEMBER EuclideanPoint
    invertMeasurement(DisparityPoint p_disparity, Extrinsics e);

    CUDA_CALLABLE_MEMBER double std_u(){ return std_u_ ; }
    CUDA_CALLABLE_MEMBER double std_v(){ return std_v_ ; }
    CUDA_CALLABLE_MEMBER double pd(){ return pd_ ; }
    CUDA_CALLABLE_MEMBER double kappa(){ return kappa_ ; }

    CUDA_CALLABLE_MEMBER double img_width(){ return img_width_ ; }
    CUDA_CALLABLE_MEMBER double img_height(){ return img_height_ ; }

protected:
    double fx_ ;
    double fy_ ;
    double u0_ ;
    double v0_ ;
    double img_width_ ;
    double img_height_ ;
    double std_u_ ;
    double std_v_ ;
    double pd_ ;
    double kappa_ ;
};


#endif // DISPARITYMEASUREMENTMODEL_CUH
