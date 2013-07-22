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

    CUDA_CALLABLE_MEMBER DisparityMeasurementModel(float fx, float fy, float u0, float v0,
                             float std_u, float std_v, float pd, float lambda);

    CUDA_CALLABLE_MEMBER DisparityPoint
    computeMeasurement(EuclideanPoint p_world);

    CUDA_CALLABLE_MEMBER DisparityPoint
    computeMeasurement(EuclideanPoint p_world, Extrinsics e);

    CUDA_CALLABLE_MEMBER EuclideanPoint
    invertMeasurement(DisparityPoint p_disparity, Extrinsics e);

    CUDA_CALLABLE_MEMBER float std_u(){ return std_u_ ; }
    CUDA_CALLABLE_MEMBER float std_v(){ return std_v_ ; }
    CUDA_CALLABLE_MEMBER float pd(){ return pd_ ; }
    CUDA_CALLABLE_MEMBER float kappa(){ return kappa_ ; }

protected:
    float fx_ ;
    float fy_ ;
    float u0_ ;
    float v0_ ;
    float img_width_ ;
    float img_height_ ;
    float std_u_ ;
    float std_v_ ;
    float pd_ ;
    float kappa_ ;
};


#endif // DISPARITYMEASUREMENTMODEL_CUH
