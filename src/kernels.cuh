#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include "types.h"
#include "disparitymeasurementmodel.cuh"

__global__ void
worldToDisparityKernel(double* x, double* y, double* z,
                       double* vx, double *vy, double* vz,
                       DisparityMeasurementModel model,
                       Extrinsics* ex, int* indices,
                       int n_particles,
                       double* u, double* v, double* d,
                       double* vu, double *vv, double* vd,
                       double* particle_pd) ;

__global__ void
disparityToWorldKernel(double* u, double* v, double* d,
                       double* vu, double* vv, double* vd,
                       DisparityMeasurementModel model,
                       Extrinsics* ex, int* indices,
                       int n_particles,
                       double* x, double* y, double* z,
                       double* vx, double* vy, double* vz) ;

__global__ void
computePdKernel(double* particle_pd, int particles_per_feature, int n_features,
                double* feature_pd) ;

__global__ void
fitGaussiansKernel(double* uArray, double* vArray, double* dArray,
                   double* vuArray, double* vvArray, double* vdArray,
                   double* weights, double *feature_weights, int nGaussians,
                   Gaussian6D* gaussians,
                   int n_particles) ;

__global__ void
computeBirthsKernel(double* u, double* v, Gaussian6D* features_birth,
                    DisparityPoint birth_mean, DisparityPoint birth_var,
                    double w0, int n_births) ;

__global__ void
computeDetectionsKernel(Gaussian6D* features, double* u, double* v, double* pd,
                        DisparityMeasurementModel model,
                        int* map_offsets,int n_features, int n_measure,
                        Gaussian6D* detections, int* measure_indices ) ;

__global__ void
updateKernel(Gaussian6D* detections,
             Gaussian6D* births, double* normalizers, int* map_offsets,
             int n_maps, int n_measure, Gaussian6D* updated,
             bool* merge_flags, double min_weight) ;

__global__ void
phdUpdateMergeKernel(Gaussian6D* updated_features,
                     Gaussian6D* mergedFeatures, int *mergedSizes,
                     bool *mergedFlags, int* map_offsets, int n_particles,
                     double min_separation) ;

//__global__ void
//recombineFeaturesKernel(Gaussian6D* features_in, Gaussian6D* features_out,
//                        int* map_offsets_in, int* map_offsets_out, int n_maps,
//                        int n_features, Gaussian6D* features_combined,
//                        int* indices_combined) ;

__global__ void
expandKernel(double* values, int n_original, int factor, double* expanded) ;

template <typename T>
__global__ void
interleaveKernel(T* items1, T* items2,
                 int* offsets_vec1, int* offsets_vec2,
                 int n_segments, int n_items,
                 T* combined, int* indices_combined) ;

__global__ void
normalizeWeightsKernel(double* weights, int n_particles) ;

__global__ void
resampleFeaturesKernel(double* u, double* v, double* d,
                       double* vu, double* vv, double* vd,
                       double* weights, double* randvals, int n_features,
                       double* u_sampled, double* v_sampled, double* d_sampled,
                       double* vu_sampled, double* vv_sampled, double* vd_sampled) ;

#endif // KERNELS_CUH
