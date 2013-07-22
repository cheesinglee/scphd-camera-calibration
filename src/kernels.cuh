#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include "types.h"

__global__ void
fitGaussiansKernel(float* uArray, float* vArray, float* dArray,
                   float* vuArray, float* vvArray, float* vdArray,
                   float* weights,int nGaussians,
                   Gaussian6D* gaussians,
                   int n_particles) ;

__global__ void
updateKernel(Gaussian6D* nondetections, Gaussian6D* detections,
             Gaussian6D* births, float* normalizers, int* map_offsets,
             int n_maps, int n_measure, Gaussian6D* updated,
             bool* merge_flags, float min_weight) ;

__global__ void
phdUpdateMergeKernel(Gaussian6D* updated_features,
                     Gaussian6D* mergedFeatures, int *mergedSizes,
                     bool *mergedFlags, int* map_offsets, int n_particles,
                     float min_separation) ;

#endif // KERNELS_CUH
