#ifndef SCPHDCAMERACALIBRATION_H
#define SCPHDCAMERACALIBRATION_H

#include "types.h"
#include "disparitymeasurementmodel.cuh"
#include "linearcvmotionmodel3d.cuh"
#include <libconfig.h++>

#include <vector>
#include <iterator>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

using namespace std ;
using namespace libconfig ;
using namespace thrust ;


class SCPHDCameraCalibration
{
public:
    SCPHDCameraCalibration(const char* config_file);

    void predict(float dt) ;
    void update(vector<float> u, vector<float> v, bool fixed_camera) ;

private:
    cudaDeviceProp cuda_dev_props_ ;
    Config config_ ;

    LinearCVMotionModel3D feature_motion_model_ ;
    DisparityMeasurementModel measurement_model_ ;

    thrust::device_vector<Extrinsics> dev_particle_states_ ;
    thrust::host_vector<float> particle_weights_ ;
    int n_particles_ ;
    int particles_per_feature_ ;

    thrust::device_vector<int> dev_particle_indices_ ;
    thrust::device_vector<int> dev_gaussian_indices_ ;
    thrust::device_vector<float> dev_feature_weights_ ;
    thrust::device_vector<float> dev_x_ ;
    thrust::device_vector<float> dev_y_ ;
    thrust::device_vector<float> dev_z_ ;
    thrust::device_vector<float> dev_vx_ ;
    thrust::device_vector<float> dev_vy_ ;
    thrust::device_vector<float> dev_vz_ ;
    thrust::host_vector<int> map_sizes_ ;

    thrust::device_vector<float> dev_u_ ;
    thrust::device_vector<float> dev_v_ ;
    thrust::device_vector<float> dev_d_ ;
    thrust::device_vector<float> dev_vu_ ;
    thrust::device_vector<float> dev_vv_ ;
    thrust::device_vector<float> dev_vd_ ;
    thrust::device_vector<Gaussian6D> dev_features_ ;

    void initializeCuda() ;

    void computeDisparityParticles() ;
    void computeEuclideanParticles() ;
    void fitGaussians() ;

    void resample() ;
};


#endif // SCPHDCAMERACALIBRATION_H
