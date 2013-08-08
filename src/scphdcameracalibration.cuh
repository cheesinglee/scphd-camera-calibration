#ifndef SCPHDCAMERACALIBRATION_H
#define SCPHDCAMERACALIBRATION_H

#include "types.h"
#include "disparitymeasurementmodel.cuh"
#include "linearcvmotionmodel3d.cuh"
#include "OrientedLinearCVMotionModel3D.cuh"

#include <libconfig.h++>
#include <matio.h>

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

    void predict(double dt) ;
    void update(vector<double> u, vector<double> v, bool fixed_camera) ;
    void writeMat(const char* filename) ;

    void resample() ;
    DisparityMeasurementModel measurement_model() const;
    void setMeasurement_model(const DisparityMeasurementModel &measurement_model);

    void transformTest() ;

    void checkStuff() ;
private:
    int verbosity_ ;

    cudaDeviceProp cuda_dev_props_ ;
    Config config_ ;

    LinearCVMotionModel3D feature_motion_model_ ;
    OrientedLinearCVMotionModel3D camera_motion_model_ ;
    DisparityMeasurementModel measurement_model_ ;

    thrust::device_vector<Extrinsics> dev_particle_states_ ;
    thrust::host_vector<double> particle_weights_ ;
    int n_particles_ ;
    int particles_per_feature_ ;

    thrust::device_vector<int> dev_particle_indices_ ;
    thrust::device_vector<int> dev_gaussian_indices_ ;

    thrust::device_vector<double> dev_feature_weights_ ;
    thrust::device_vector<double> dev_x_ ;
    thrust::device_vector<double> dev_y_ ;
    thrust::device_vector<double> dev_z_ ;
    thrust::device_vector<double> dev_vx_ ;
    thrust::device_vector<double> dev_vy_ ;
    thrust::device_vector<double> dev_vz_ ;

    thrust::device_vector<double> dev_u_ ;
    thrust::device_vector<double> dev_v_ ;
    thrust::device_vector<double> dev_d_ ;
    thrust::device_vector<double> dev_vu_ ;
    thrust::device_vector<double> dev_vv_ ;
    thrust::device_vector<double> dev_vd_ ;

    thrust::device_vector<double> dev_pd_ ;
    thrust::device_vector<double> dev_particle_pd_ ;
    thrust::device_vector<double> dev_particle_weights_ ;
    thrust::device_vector<double> dev_particle_weights_nondetect_ ;

    thrust::device_vector<double> dev_u_nondetect_ ;
    thrust::device_vector<double> dev_v_nondetect_ ;
    thrust::device_vector<double> dev_d_nondetect_ ;
    thrust::device_vector<double> dev_vu_nondetect_ ;
    thrust::device_vector<double> dev_vv_nondetect_ ;
    thrust::device_vector<double> dev_vd_nondetect_ ;
    thrust::device_vector<double> dev_feature_weights_nondetect_ ;
    thrust::device_vector<int> dev_indices_nondetect_ ;

    thrust::device_vector<Gaussian6D> dev_features_ ;
    thrust::host_vector<int> gaussian_indices_predicted_ ;

    thrust::device_vector<Gaussian6D> dev_features_updated_ ;
    thrust::host_vector<int> gaussian_indices_updated_ ;
    thrust::device_vector<Gaussian6D> dev_features_predicted_ ;

    thrust::device_vector<Gaussian6D> dev_features_out_of_range_ ;
    thrust::device_vector<int> dev_indices_out_of_range_ ;

    void initializeCuda() ;

    void computeDisparityParticles(bool fixed_camera) ;
    void computeEuclideanParticles(bool fixed_camera) ;

    void computeNonDetections() ;
    void recombineNonDetections() ;

    double computeNeff();
    void fitGaussians() ;

    void separateOutOfRange() ;
    void recombineFeatures() ;

    device_vector<int> computeMapOffsets(device_vector<int> indices) ;

    thrust::device_vector<Gaussian6D> computeBirths(vector<double> u, vector <double> v) ;

    matvar_t* writeGaussianMixtureMatVar(host_vector<Gaussian6D> gm, host_vector<int> indices, const char *varname);
    matvar_t *writeGaussianMixtureMatVar(host_vector<Gaussian6D> gm, host_vector<int> indices, const char *varname, int idx);
};


#endif // SCPHDCAMERACALIBRATION_H
