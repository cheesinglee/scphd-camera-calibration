#include "kernels.cuh"
#include "device_math.cuh"

#include <stdio.h>

__global__ void
worldToDisparityKernel(double* x, double* y, double* z,
                       double* vx, double *vy, double* vz,
                       DisparityMeasurementModel model,
                       Extrinsics* ex, int* indices,
                       int n_particles,
                       double* u, double* v, double* d,
                       double* vu, double *vv, double* vd,
                       double* particle_pd){

    int tid = threadIdx.x + blockIdx.x*blockDim.x ;
    if (tid < n_particles){
        EuclideanPoint p_world ;
        p_world.x = x[tid] ;
        p_world.y = y[tid] ;
        p_world.z = z[tid] ;
        p_world.vx = vx[tid] ;
        p_world.vy = vy[tid] ;
        p_world.vz = vz[tid] ;

        int cam_idx = indices[tid] ;
        Extrinsics e = ex[cam_idx] ;

        DisparityPoint p_disparity = model.computeMeasurement(p_world,e) ;
        u[tid] = p_disparity.u ;
        v[tid] = p_disparity.v ;
        d[tid] = p_disparity.d ;
        vu[tid] = p_disparity.vu ;
        vv[tid] = p_disparity.vv ;
        vd[tid] = p_disparity.vd ;

        bool in_range = ( p_disparity.u >= 0 ) &&
                ( p_disparity.u <= model.img_width() ) &&
                ( p_disparity.v >= 0 ) &&
                ( p_disparity.v <= model.img_height() ) &&
                ( p_disparity.d >= 0 ) ;
        if (in_range)
            particle_pd[tid] = model.pd() ;
        else
            particle_pd[tid] = 0 ;
    }
}

__global__ void
disparityToWorldKernel(double* u, double* v, double* d,
                       double* vu, double* vv, double* vd,
                       DisparityMeasurementModel model,
                       Extrinsics* ex,
                       int* indices, int n_particles,
                       double* x, double* y, double* z,
                       double* vx, double* vy, double* vz)
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x ;
    if (tid < n_particles){
        DisparityPoint p_disparity ;
        p_disparity.u = u[tid] ;
        p_disparity.v = v[tid] ;
        p_disparity.d = d[tid] ;
        p_disparity.vu = vu[tid] ;
        p_disparity.vv = vv[tid] ;
        p_disparity.vd = vd[tid] ;

        int cam_idx = indices[tid] ;
        Extrinsics e = ex[cam_idx] ;

        EuclideanPoint p_world = model.invertMeasurement(p_disparity,e) ;

        x[tid] = p_world.x ;
        y[tid] = p_world.y ;
        z[tid] = p_world.z ;
        vx[tid] = p_world.vx ;
        vy[tid] = p_world.vy ;
        vz[tid] = p_world.vz ;
    }
}

__global__ void
computePdKernel(double* particle_pd, int particles_per_feature, int n_features,
                double* feature_pd)
{
    __shared__ double shmem[256] ;
    for ( int n = blockIdx.x ; n < n_features ;n+= gridDim.x ){
        int offset = n*particles_per_feature ;
        double val = 0 ;
        for ( int i = offset+threadIdx.x ; i < offset + particles_per_feature ; i+= blockDim.x ){
            val += particle_pd[i] ;
        }
        sumByReduction(shmem,val,threadIdx.x);

        if ( threadIdx.x == 0)
            feature_pd[n] = shmem[0]/particles_per_feature ;
        __syncthreads() ;
    }
}


__global__ void
fitGaussiansKernel(double* uArray, double* vArray, double* dArray,
                   double* vuArray, double* vvArray, double* vdArray,
                   double* weights, double* feature_weights, int nGaussians,
                   Gaussian6D* gaussians,
                   int n_particles){
    int tid = threadIdx.x ;
    __shared__ double sdata[256] ;
    for (int i = blockIdx.x ; i < nGaussians ; i+=gridDim.x){
        int offset = i*n_particles ;
        double val = 0 ;

        // compute mean u
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += uArray[offset+j]*weights[offset+j] ;
        }
        sumByReduction(sdata,val,tid);
        double uMean = sdata[0] ;
        __syncthreads() ;

        // compute mean v
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += vArray[offset+j]*weights[offset+j] ;
        }
        sumByReduction(sdata,val,tid);
        double vMean = sdata[0] ;
        __syncthreads() ;

        // compute mean d
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += dArray[offset+j]*weights[offset+j] ;
        }
        sumByReduction(sdata,val,tid);
        double dMean = sdata[0] ;
        __syncthreads() ;

        // compute mean vu
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += vuArray[offset+j]*weights[offset+j] ;
        }
        sumByReduction(sdata,val,tid);
        double vuMean = sdata[0] ;
        __syncthreads() ;

        // compute mean vv
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += vvArray[offset+j]*weights[offset+j] ;
        }
        sumByReduction(sdata,val,tid);
        double vvMean = sdata[0] ;
        __syncthreads() ;

        // compute mean vd
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += vdArray[offset+j]*weights[offset+j] ;
        }
        sumByReduction(sdata,val,tid);
        double vdMean = sdata[0] ;
        __syncthreads() ;


        // write means to output
        if (tid == 0){
            gaussians[i].weight = feature_weights[i] ;
            gaussians[i].mean[0] = uMean ;
            gaussians[i].mean[1] = vMean ;
            gaussians[i].mean[2] = dMean ;
            gaussians[i].mean[3] = vuMean ;
            gaussians[i].mean[4] = vvMean ;
            gaussians[i].mean[5] = vdMean ;
        }

        // compute sum of particle weights
        val = 0 ;
        for (int j = tid ; j < n_particles ; j += blockDim.x){
            val += weights[offset+j] ;
        }
        sumByReduction(sdata,val,tid);
        double weight = sdata[0] ;
        __syncthreads() ;

        // compute normalizer for covariance
        val = 0 ;
        for (int j = tid; j < n_particles ; j+=blockDim.x){
            val += weights[offset+j]*weights[offset+j] ;
        }
        sumByReduction(sdata,val,tid);
        double sum_of_squares = sdata[0] ;
        double cov_normalizer = weight/(weight*weight + sum_of_squares) ;

        // covariance term u-u
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += pow(uArray[offset+j]-uMean,2)*weights[offset+j] ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0)
            gaussians[i].cov[0] = sdata[0]/cov_normalizer ;
        __syncthreads() ;

        // covariance term v-v
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += pow(vArray[offset+j]-vMean,2)*weights[offset+j] ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0)
            gaussians[i].cov[7] = sdata[0]/cov_normalizer ;
        __syncthreads() ;

        // covariance term d-d
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += pow(dArray[offset+j]-dMean,2)*weights[offset+j] ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0)
            gaussians[i].cov[14] = sdata[0]/cov_normalizer ;
        __syncthreads() ;

        // covariance term vu-vu
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += pow(vuArray[offset+j]-vuMean,2)*weights[offset+j] ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0)
            gaussians[i].cov[21] = sdata[0]/(cov_normalizer) ;
        __syncthreads() ;

        // covariance term vv-vv
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += pow(vvArray[offset+j]-vvMean,2)*weights[offset+j] ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0)
            gaussians[i].cov[28] = sdata[0]/cov_normalizer ;
        __syncthreads() ;

        // covariance term vd-vd
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += pow(vdArray[offset+j]-vdMean,2)*weights[offset+j] ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0)
            gaussians[i].cov[35] = sdata[0]/cov_normalizer ;
        __syncthreads() ;

        // covariance term u-v
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += (uArray[offset+j]-uMean)*(vArray[offset+j]-vMean)*weights[offset+j] ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0){
            gaussians[i].cov[1] = sdata[0]/cov_normalizer ;
            gaussians[i].cov[6] = gaussians[i].cov[1] ;
        }
        __syncthreads() ;

        // covariance term u-d
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += (uArray[offset+j]-uMean)*(dArray[offset+j]-dMean)*weights[offset+j] ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0){
            gaussians[i].cov[2] = sdata[0]/cov_normalizer ;
            gaussians[i].cov[12] = gaussians[i].cov[2] ;
        }
        __syncthreads() ;

        // covariance term u-vu
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += (uArray[offset+j]-uMean)*(vuArray[offset+j]-vuMean)*weights[offset+j] ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0){
            gaussians[i].cov[3] = sdata[0]/cov_normalizer ;
            gaussians[i].cov[18] = gaussians[i].cov[3] ;
        }
        __syncthreads() ;

        // covariance term u-vv
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += (uArray[offset+j]-uMean)*(vvArray[offset+j]-vvMean)*weights[offset+j] ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0){
            gaussians[i].cov[4] = sdata[0]/cov_normalizer ;
            gaussians[i].cov[24] = gaussians[i].cov[4] ;
        }
        __syncthreads() ;

        // covariance term u-vd
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += (uArray[offset+j]-uMean)*(vdArray[offset+j]-vdMean)*weights[offset+j] ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0){
            gaussians[i].cov[5] = sdata[0]/cov_normalizer ;
            gaussians[i].cov[30] = gaussians[i].cov[5] ;
        }
        __syncthreads() ;

        // covariance term v-d
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += (vArray[offset+j]-vMean)*(dArray[offset+j]-dMean)*weights[offset+j] ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0){
            gaussians[i].cov[8] = sdata[0]/cov_normalizer ;
            gaussians[i].cov[13] = gaussians[i].cov[8] ;
        }
        __syncthreads() ;

        // covariance term v-vu
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += (vArray[offset+j]-vMean)*(vuArray[offset+j]-vuMean)*weights[offset+j] ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0){
            gaussians[i].cov[9] = sdata[0]/cov_normalizer ;
            gaussians[i].cov[19] = gaussians[i].cov[9] ;
        }
        __syncthreads() ;

        // covariance term v-vv
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += (vArray[offset+j]-vMean)*(vvArray[offset+j]-vvMean)*weights[offset+j] ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0){
            gaussians[i].cov[10] = sdata[0]/cov_normalizer ;
            gaussians[i].cov[25] = gaussians[i].cov[10] ;
        }
        __syncthreads() ;

        // covariance term v-vd
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += (vArray[offset+j]-vMean)*(vdArray[offset+j]-vdMean)*weights[offset+j] ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0){
            gaussians[i].cov[11] = sdata[0]/cov_normalizer ;
            gaussians[i].cov[31] = gaussians[i].cov[11] ;
        }
        __syncthreads() ;

        // covariance term d-vu
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += (dArray[offset+j]-dMean)*(vuArray[offset+j]-vuMean)*weights[offset+j] ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0){
            gaussians[i].cov[15] = sdata[0]/cov_normalizer ;
            gaussians[i].cov[20] = gaussians[i].cov[15] ;
        }
        __syncthreads() ;

        // covariance term d-vv
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += (dArray[offset+j]-dMean)*(vvArray[offset+j]-vvMean)*weights[offset+j] ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0){
            gaussians[i].cov[16] = sdata[0]/cov_normalizer ;
            gaussians[i].cov[26] = gaussians[i].cov[16] ;
        }
        __syncthreads() ;

        // covariance term d-vd
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += (dArray[offset+j]-dMean)*(vdArray[offset+j]-vdMean)*weights[offset+j] ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0){
            gaussians[i].cov[17] = sdata[0]/cov_normalizer ;
            gaussians[i].cov[32] = gaussians[i].cov[17] ;
        }
        __syncthreads() ;

        // covariance term vu-vv
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += (vuArray[offset+j]-vuMean)*(vvArray[offset+j]-vvMean)*weights[offset+j] ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0){
            gaussians[i].cov[22] = sdata[0]/cov_normalizer ;
            gaussians[i].cov[27] = gaussians[i].cov[22] ;
        }
        __syncthreads() ;

        // covariance term vu-vd
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += (vuArray[offset+j]-vuMean)*(vdArray[offset+j]-vdMean)*weights[offset+j] ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0){
            gaussians[i].cov[23] = sdata[0]/cov_normalizer ;
            gaussians[i].cov[33] = gaussians[i].cov[23] ;
        }
        __syncthreads() ;

        // covariance term vv-vd
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += (vvArray[offset+j]-vvMean)*(vdArray[offset+j]-vdMean)*weights[offset+j] ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0){
            gaussians[i].cov[29] = sdata[0]/cov_normalizer ;
            gaussians[i].cov[34] = gaussians[i].cov[29] ;
        }
        __syncthreads() ;
    }
}

__global__ void
computeBirthsKernel(double* u, double* v, Gaussian6D* features_birth,
                    DisparityPoint birth_mean, DisparityPoint birth_var,
                    double w0, int n_births)
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x ;
    if ( tid < n_births ){
        Gaussian6D f ;
        f.mean[0] = u[tid] ;
        f.mean[1] = v[tid] ;
        f.mean[2] = birth_mean.d ;
        f.mean[3] = birth_mean.vu ;
        f.mean[4] = birth_mean.vv ;
        f.mean[5] = birth_mean.vd ;

        for ( int i = 0 ; i < 36 ; i++)
            f.cov[i] = 0 ;
        f.cov[0] = birth_var.u ;
        f.cov[7] = birth_var.v ;
        f.cov[14] = birth_var.d ;
        f.cov[21] = birth_var.vu ;
        f.cov[28] = birth_var.vv ;
        f.cov[35] = birth_var.vd ;

        f.weight = safeLog(w0) ;

        features_birth[tid] = f ;
    }
}

__global__ void
computeDetectionsKernel(Gaussian6D* features, double* u, double* v, double* pd,
                        DisparityMeasurementModel model,
                        int* map_offsets,int n_features, int n_measure,
                        Gaussian6D* detections, int* measure_indices )
{
    // compute indices
    int idx_feature = threadIdx.x + blockIdx.x*blockDim.x ;
    int idx_measure = blockIdx.y ;

    if ( idx_feature < n_features ){
        // loop through map offsets until we run into one that is greater than
        // the feature index. this corresponds to the thread's map.
        int map_idx = 0 ;
        while (idx_feature >= map_offsets[map_idx+1])
            map_idx++ ;

        // more indices
        int offset = map_offsets[map_idx] ;
        int map_size = map_offsets[map_idx+1] - offset ;
        int idx_detect = (idx_feature-offset)
                + map_size*idx_measure
                + n_measure*offset ;

        Gaussian6D f = features[idx_feature] ;
        Gaussian6D f_update ;
        double* p = f.cov ;

        double var_u = model.std_u()*model.std_u() ;
        double var_v = model.std_v()*model.std_v() ;

        // innovation vector
        double innov[2] ;
        innov[0] = u[idx_measure] - f.mean[0] ;
        innov[1] = v[idx_measure] - f.mean[1] ;

        // Innovation covariance
        double sigma[4] ;
        sigma[0] = p[0] + var_u;
        sigma[1] = p[1];
        sigma[2] = p[6];
        sigma[3] = p[7] + var_v;

        // enforce symmetry
        sigma[1] = (sigma[1]+sigma[2])/2 ;
        sigma[2] = sigma[1] ;

        // inverse sigma
        double s[4] ;
        double det = sigma[0]*sigma[3] - sigma[1]*sigma[2] ;
        s[0] = sigma[3]/det ;
        s[1] = -sigma[1]/det ;
        s[2] = -sigma[2]/det ;
        s[3] = sigma[0]/det ;

        // measurement likelihood
        double dist = innov[0]*innov[0]*s[0] +
                innov[0]*innov[1]*(s[1] + s[2]) +
                innov[1]*innov[1]*s[3] ;
        f_update.weight = safeLog(pd[idx_feature])
                + safeLog(f.weight)
                - 0.5*dist
                - safeLog(2*M_PI)
                - 0.5*safeLog(det) ;

        // Kalman gain K = PH'/S
        double K[12] ;
        K[0] = p[0] * s[0] + p[6] * s[1];
        K[1] = p[1] * s[0] + p[7] * s[1];
        K[2] = p[2] * s[0] + p[8] * s[1];
        K[3] = p[3] * s[0] + p[9] * s[1];
        K[4] = p[4] * s[0] + p[10] * s[1];
        K[5] = p[5] * s[0] + p[11] * s[1];
        K[6] = p[0] * s[2] + p[6] * s[3];
        K[7] = p[1] * s[2] + p[7] * s[3];
        K[8] = p[2] * s[2] + p[8] * s[3];
        K[9] = p[3] * s[2] + p[9] * s[3];
        K[10] = p[4] * s[2] + p[10] * s[3];
        K[11] = p[5] * s[2] + p[11] * s[3];

        // updated mean x = x + K*innov
        f_update.mean[0] = f.mean[0] + innov[0]*K[0] + innov[1]*K[6] ;
        f_update.mean[1] = f.mean[1] + innov[0]*K[1] + innov[1]*K[7] ;
        f_update.mean[2] = f.mean[2] + innov[0]*K[2] + innov[1]*K[8] ;
        f_update.mean[3] = f.mean[3] + innov[0]*K[3] + innov[1]*K[9] ;
        f_update.mean[4] = f.mean[4] + innov[0]*K[4] + innov[1]*K[10] ;
        f_update.mean[5] = f.mean[5] + innov[0]*K[5] + innov[1]*K[11] ;

        // updated covariance P = IKH*P/IKH' + KRK'

        f_update.cov[0] = (1 - K[0]) * (p[0] * (1 - K[0]) - p[6] * K[6]) - K[6] * (p[1] * (1 - K[0]) - p[7] * K[6]) + var_u *  K[0]*K[0] + var_v * K[6]*K[6];
        f_update.cov[1] = -K[1] * (p[0] * (1 - K[0]) - p[6] * K[6]) + (1 - K[7]) * (p[1] * (1 - K[0]) - p[7] * K[6]) + K[0] * var_u * K[1] + K[6] * var_v * K[7];
        f_update.cov[2] = -K[2] * (p[0] * (1 - K[0]) - p[6] * K[6]) - K[8] * (p[1] * (1 - K[0]) - p[7] * K[6]) + p[2] * (1 - K[0]) - p[8] * K[6] + K[0] * var_u * K[2] + K[6] * var_v * K[8];
        f_update.cov[3] = -K[3] * (p[0] * (1 - K[0]) - p[6] * K[6]) - K[9] * (p[1] * (1 - K[0]) - p[7] * K[6]) + p[3] * (1 - K[0]) - p[9] * K[6] + K[0] * var_u * K[3] + K[6] * var_v * K[9];
        f_update.cov[4] = -K[4] * (p[0] * (1 - K[0]) - p[6] * K[6]) - K[10] * (p[1] * (1 - K[0]) - p[7] * K[6]) + p[4] * (1 - K[0]) - p[10] * K[6] + K[0] * var_u * K[4] + K[6] * var_v * K[10];
        f_update.cov[5] = -K[5] * (p[0] * (1 - K[0]) - p[6] * K[6]) - K[11] * (p[1] * (1 - K[0]) - p[7] * K[6]) + p[5] * (1 - K[0]) - p[11] * K[6] + K[0] * var_u * K[5] + K[6] * var_v * K[11];
        f_update.cov[6] = (1 - K[0]) * (-p[0] * K[1] + p[6] * (1 - K[7])) - K[6] * (-p[1] * K[1] + p[7] * (1 - K[7])) + K[0] * var_u * K[1] + K[6] * var_v * K[7];
        f_update.cov[7] = -K[1] * (-p[0] * K[1] + p[6] * (1 - K[7])) + (1 - K[7]) * (-p[1] * K[1] + p[7] * (1 - K[7])) + var_u *  K[1]*K[1] + var_v *  K[7]*K[7];
        f_update.cov[8] = -K[2] * (-p[0] * K[1] + p[6] * (1 - K[7])) - K[8] * (-p[1] * K[1] + p[7] * (1 - K[7])) - p[2] * K[1] + p[8] * (1 - K[7]) + K[1] * var_u * K[2] + K[7] * var_v * K[8];
        f_update.cov[9] = -K[3] * (-p[0] * K[1] + p[6] * (1 - K[7])) - K[9] * (-p[1] * K[1] + p[7] * (1 - K[7])) - p[3] * K[1] + p[9] * (1 - K[7]) + K[1] * var_u * K[3] + K[7] * var_v * K[9];
        f_update.cov[10] = -K[4] * (-p[0] * K[1] + p[6] * (1 - K[7])) - K[10] * (-p[1] * K[1] + p[7] * (1 - K[7])) - p[4] * K[1] + p[10] * (1 - K[7]) + K[1] * var_u * K[4] + K[7] * var_v * K[10];
        f_update.cov[11] = -K[5] * (-p[0] * K[1] + p[6] * (1 - K[7])) - K[11] * (-p[1] * K[1] + p[7] * (1 - K[7])) - p[5] * K[1] + p[11] * (1 - K[7]) + K[1] * var_u * K[5] + K[7] * var_v * K[11];
        f_update.cov[12] = (1 - K[0]) * (-p[0] * K[2] - p[6] * K[8] + p[12]) - K[6] * (-p[1] * K[2] - p[7] * K[8] + p[13]) + K[0] * var_u * K[2] + K[6] * var_v * K[8];
        f_update.cov[13] = -K[1] * (-p[0] * K[2] - p[6] * K[8] + p[12]) + (1 - K[7]) * (-p[1] * K[2] - p[7] * K[8] + p[13]) + K[1] * var_u * K[2] + K[7] * var_v * K[8];
        f_update.cov[14] = -K[2] * (-p[0] * K[2] - p[6] * K[8] + p[12]) - K[8] * (-p[1] * K[2] - p[7] * K[8] + p[13]) - p[2] * K[2] - p[8] * K[8] + p[14] + var_u * K[2]*K[2] + var_v * K[8]*K[8];
        f_update.cov[15] = -K[3] * (-p[0] * K[2] - p[6] * K[8] + p[12]) - K[9] * (-p[1] * K[2] - p[7] * K[8] + p[13]) - p[3] * K[2] - p[9] * K[8] + p[15] + K[2] * var_u * K[3] + K[8] * var_v * K[9];
        f_update.cov[16] = -K[4] * (-p[0] * K[2] - p[6] * K[8] + p[12]) - K[10] * (-p[1] * K[2] - p[7] * K[8] + p[13]) - p[4] * K[2] - p[10] * K[8] + p[16] + K[2] * var_u * K[4] + K[8] * var_v * K[10];
        f_update.cov[17] = -K[5] * (-p[0] * K[2] - p[6] * K[8] + p[12]) - K[11] * (-p[1] * K[2] - p[7] * K[8] + p[13]) - p[5] * K[2] - p[11] * K[8] + p[17] + K[2] * var_u * K[5] + K[8] * var_v * K[11];
        f_update.cov[18] = (1 - K[0]) * (-p[0] * K[3] - p[6] * K[9] + p[18]) - K[6] * (-p[1] * K[3] - p[7] * K[9] + p[19]) + K[0] * var_u * K[3] + K[6] * var_v * K[9];
        f_update.cov[19] = -K[1] * (-p[0] * K[3] - p[6] * K[9] + p[18]) + (1 - K[7]) * (-p[1] * K[3] - p[7] * K[9] + p[19]) + K[1] * var_u * K[3] + K[7] * var_v * K[9];
        f_update.cov[20] = -K[2] * (-p[0] * K[3] - p[6] * K[9] + p[18]) - K[8] * (-p[1] * K[3] - p[7] * K[9] + p[19]) - p[2] * K[3] - p[8] * K[9] + p[20] + K[2] * var_u * K[3] + K[8] * var_v * K[9];
        f_update.cov[21] = -K[3] * (-p[0] * K[3] - p[6] * K[9] + p[18]) - K[9] * (-p[1] * K[3] - p[7] * K[9] + p[19]) - p[3] * K[3] - p[9] * K[9] + p[21] + var_u *  K[3]*K[3] + var_v * K[9]*K[9];
        f_update.cov[22] = -K[4] * (-p[0] * K[3] - p[6] * K[9] + p[18]) - K[10] * (-p[1] * K[3] - p[7] * K[9] + p[19]) - p[4] * K[3] - p[10] * K[9] + p[22] + K[3] * var_u * K[4] + K[9] * var_v * K[10];
        f_update.cov[23] = -K[5] * (-p[0] * K[3] - p[6] * K[9] + p[18]) - K[11] * (-p[1] * K[3] - p[7] * K[9] + p[19]) - p[5] * K[3] - p[11] * K[9] + p[23] + K[3] * var_u * K[5] + K[9] * var_v * K[11];
        f_update.cov[24] = (1 - K[0]) * (-p[0] * K[4] - p[6] * K[10] + p[24]) - K[6] * (-p[1] * K[4] - p[7] * K[10] + p[25]) + K[0] * var_u * K[4] + K[6] * var_v * K[10];
        f_update.cov[25] = -K[1] * (-p[0] * K[4] - p[6] * K[10] + p[24]) + (1 - K[7]) * (-p[1] * K[4] - p[7] * K[10] + p[25]) + K[1] * var_u * K[4] + K[7] * var_v * K[10];
        f_update.cov[26] = -K[2] * (-p[0] * K[4] - p[6] * K[10] + p[24]) - K[8] * (-p[1] * K[4] - p[7] * K[10] + p[25]) - p[2] * K[4] - p[8] * K[10] + p[26] + K[2] * var_u * K[4] + K[8] * var_v * K[10];
        f_update.cov[27] = -K[3] * (-p[0] * K[4] - p[6] * K[10] + p[24]) - K[9] * (-p[1] * K[4] - p[7] * K[10] + p[25]) - p[3] * K[4] - p[9] * K[10] + p[27] + K[3] * var_u * K[4] + K[9] * var_v * K[10];
        f_update.cov[28] = -K[4] * (-p[0] * K[4] - p[6] * K[10] + p[24]) - K[10] * (-p[1] * K[4] - p[7] * K[10] + p[25]) - p[4] * K[4] - p[10] * K[10] + p[28] + var_u * K[4]*K[4] + var_v *  K[10]*K[10];
        f_update.cov[29] = -K[5] * (-p[0] * K[4] - p[6] * K[10] + p[24]) - K[11] * (-p[1] * K[4] - p[7] * K[10] + p[25]) - p[5] * K[4] - p[11] * K[10] + p[29] + K[4] * var_u * K[5] + K[10] * var_v * K[11];
        f_update.cov[30] = (1 - K[0]) * (-p[0] * K[5] - p[6] * K[11] + p[30]) - K[6] * (-p[1] * K[5] - p[7] * K[11] + p[31]) + K[0] * var_u * K[5] + K[6] * var_v * K[11];
        f_update.cov[31] = -K[1] * (-p[0] * K[5] - p[6] * K[11] + p[30]) + (1 - K[7]) * (-p[1] * K[5] - p[7] * K[11] + p[31]) + K[1] * var_u * K[5] + K[7] * var_v * K[11];
        f_update.cov[32] = -K[2] * (-p[0] * K[5] - p[6] * K[11] + p[30]) - K[8] * (-p[1] * K[5] - p[7] * K[11] + p[31]) - p[2] * K[5] - p[8] * K[11] + p[32] + K[2] * var_u * K[5] + K[8] * var_v * K[11];
        f_update.cov[33] = -K[3] * (-p[0] * K[5] - p[6] * K[11] + p[30]) - K[9] * (-p[1] * K[5] - p[7] * K[11] + p[31]) - p[3] * K[5] - p[9] * K[11] + p[33] + K[3] * var_u * K[5] + K[9] * var_v * K[11];
        f_update.cov[34] = -K[4] * (-p[0] * K[5] - p[6] * K[11] + p[30]) - K[10] * (-p[1] * K[5] - p[7] * K[11] + p[31]) - p[4] * K[5] - p[10] * K[11] + p[34] + K[4] * var_u * K[5] + K[10] * var_v * K[11];
        f_update.cov[35] = -K[5] * (-p[0] * K[5] - p[6] * K[11] + p[30]) - K[11] * (-p[1] * K[5] - p[7] * K[11] + p[31]) - p[5] * K[5] - p[11] * K[11] + p[35] + var_u * K[5]*K[5] + var_v *  K[11]*K[11];

        if (!isPosDef(f_update))
            f_update.weight = safeLog(0) ;

        detections[idx_detect] = f_update ;
        measure_indices[idx_detect] = idx_measure ;
    }
}


/// kernel to pack detection terms and birth terms into a single
/// data structure, while normalizing weights.
__global__ void
updateKernel(Gaussian6D* detections, Gaussian6D* births,
             double* normalizers, int* map_offsets,
             int n_maps, int n_measure, Gaussian6D* updated,
             bool* merge_flags, double min_weight){
    int tid = threadIdx.x ;
    int idx_map = blockIdx.x ;

//    int n_total_features = map_offsets[n_maps] ;
    int predict_offset = map_offsets[idx_map] ;
    int update_offset = map_offsets[idx_map]*n_measure +
            idx_map*n_measure;
    int n_features = map_offsets[idx_map+1] - map_offsets[idx_map] ;
    int n_update = n_features*n_measure + n_measure ;
    for ( int j = tid ; j < n_update ; j+= blockDim.x ){
        int idx_update = update_offset + j ;
        Gaussian6D feature ;
        if ( j < n_features*n_measure ){
            // detection term
            int idx_measure = floor(double(j)/n_features) ;
            int idx_feature = fmod(double(j),n_features) ;
            int idx_detect = n_measure*predict_offset
                    + idx_measure*n_features + idx_feature ;
            feature = detections[idx_detect] ;

            int idx_normalizer = idx_measure + idx_map*n_measure ;
            feature.weight -= normalizers[idx_normalizer] ;
            feature.weight = exp(feature.weight) ;
        }
        else if (j >= n_features*n_measure ){
            // birth term
            int idx = j - (n_features*n_measure) ;
            feature = births[idx] ;

            int idx_measure = idx ;
            int idx_normalizer = idx_measure + idx_map*n_measure ;
            feature.weight -= normalizers[idx_normalizer] ;
            feature.weight = exp(feature.weight) ;
        }
        updated[idx_update] = feature ;

        if (feature.weight < min_weight)
            merge_flags[idx_update] = true ;
        else if(!isPosDef(feature)){
            printf("deactivating non PD feature (weight = %f)\n",feature.weight) ;
            merge_flags[idx_update] = true ;
        }
        else
            merge_flags[idx_update] = false ;

    }
}


__global__ void
phdUpdateMergeKernel(Gaussian6D* updated_features,
                     Gaussian6D* mergedFeatures, int* mergedSizes,
                     bool* mergedFlags, int* map_offsets, int n_particles,
                     double min_separation)
{
    __shared__ Gaussian6D maxFeature ;
    __shared__ Gaussian6D mergedFeature ;
    __shared__ double sdata[256] ;
    __shared__ int mergedSize ;
    __shared__ int update_offset ;
    __shared__ int n_update ;
    int tid = threadIdx.x ;
    double dist ;
    Gaussian6D feature = updated_features[tid];
    int dims = 6 ;

    // loop over particles
    for ( int p = 0 ; p < n_particles ; p += gridDim.x )
    {
        int map_idx = p + blockIdx.x ;
        if ( map_idx < n_particles )
        {
            // initialize shared vars
            if ( tid == 0)
            {
                update_offset = map_offsets[map_idx] ;
                n_update = map_offsets[map_idx+1] - map_offsets[map_idx] ;
                mergedSize = 0 ;
            }
            __syncthreads() ;
            while(true)
            {
                // initialize the output values to defaults
                if ( tid == 0 )
                {
                    maxFeature.weight = -1 ;
                    clearGaussian(mergedFeature) ;
                }
                sdata[tid] = -1 ;
                __syncthreads() ;

                // find the maximum feature with parallel reduction
                for ( int i = update_offset ; i < update_offset + n_update ; i += blockDim.x)
                {
                    int idx = i + tid ;
                    if ( idx < (update_offset + n_update) )
                    {
                        if( !mergedFlags[idx] )
                        {
                            if (sdata[tid] == -1 ||
                                updated_features[(unsigned int)sdata[tid]].weight < updated_features[idx].weight )
                            {
                                sdata[tid] = idx ;
                            }
                        }
                    }
                }
                __syncthreads() ;

                for ( int s = blockDim.x/2 ; s > 0 ; s >>= 1 )
                {
                    if ( tid < s )
                    {
                        if ( sdata[tid] == -1 )
                            sdata[tid] = sdata[tid+s] ;
                        else if ( sdata[tid+s] >= 0 )
                        {
                            if(updated_features[(unsigned int)sdata[tid]].weight <
                            updated_features[(unsigned int)sdata[tid+s]].weight )
                            {
                                sdata[tid] = sdata[tid+s] ;
                            }
                        }

                    }
                    __syncthreads() ;
                }
                if ( sdata[0] == -1 || maxFeature.weight == 0 )
                    break ;
                else if(tid == 0){
                    maxFeature = updated_features[ (unsigned int)sdata[0] ] ;
                }
                __syncthreads() ;

//                if (tid == 0)
//                    printf("map %d: maxFeature idx = %d\n",map_idx,(unsigned int)sdata[0]) ;

                // find features to merge with max feature
                double sval0 = 0 ;
//                double sval1 = 0 ;
//                double sval2 = 0 ;
                clearGaussian(feature) ;
                for ( int i = update_offset ; i < update_offset + n_update ; i += blockDim.x )
                {
                    int idx = tid + i ;
                    if ( idx < update_offset+n_update )
                    {
                        if ( !mergedFlags[idx] )
                        {
                            dist = computeMahalDist(maxFeature, updated_features[idx]) ;
//                            if ( dev_config.distanceMetric == 0 )
//                                dist = computeMahalDist(maxFeature, updated_features[idx]) ;
//                            else if ( dev_config.distanceMetric == 1)
//                                dist = computeHellingerDist(maxFeature, updated_features[idx]) ;
//                            printf("map %d: distance thread %d = %f\n",map_idx,idx,dist) ;
                            if ( dist < min_separation )
                            {
                                feature.weight += updated_features[idx].weight ;
                                for ( int j = 0 ; j < dims ; j++ )
                                    feature.mean[j] += updated_features[idx].weight*updated_features[idx].mean[j] ;
                            }
                        }
                    }
                }
                // merge means and weights
                sval0 = feature.weight ;
                sumByReduction(sdata, sval0, tid) ;
                if ( tid == 0 )
                    mergedFeature.weight = sdata[0] ;
                __syncthreads() ;
                if ( mergedFeature.weight == 0 )
                    break ;
                for ( int j = 0 ; j < dims ; j++ )
                {
                    sval0 = feature.mean[j] ;
                    sumByReduction(sdata,sval0,tid);
                    if( tid == 0 )
                        mergedFeature.mean[j] = sdata[0]/mergedFeature.weight ;
                    __syncthreads() ;
                }

                // merge the covariances
                sval0 = 0 ;
//                sval1 = 0 ;
//                sval2 = 0 ;
                clearGaussian(feature) ;
                for ( int i = update_offset ; i < update_offset+n_update ; i += blockDim.x )
                {
                    int idx = tid + i ;
                    if ( idx < update_offset+n_update )
                    {
                        if (!mergedFlags[idx])
                        {
                            dist = computeMahalDist(maxFeature, updated_features[idx]) ;
//                            if ( dev_config.distanceMetric == 0 )
//                                dist = computeMahalDist(maxFeature, updated_features[idx]) ;
//                            else if ( dev_config.distanceMetric == 1)
//                                dist = computeHellingerDist(maxFeature, updated_features[idx]) ;
                            if ( dist < min_separation )
                            {
                                // use the mean of the local gaussian variable
                                // to store the innovation vector
                                for (int j = 0 ; j < dims ; j++)
                                {
                                    feature.mean[j] = mergedFeature.mean[j]
                                            - updated_features[idx].mean[j] ;
                                }
                                for (int j = 0 ; j < dims ; j++ )
                                {
                                    double outer = feature.mean[j] ;
                                    for ( int k = 0 ; k < dims ; k++)
                                    {
                                        double inner = feature.mean[k] ;
                                        feature.cov[j*dims+k] +=
                                                updated_features[idx].weight*
                                                (updated_features[idx].cov[j*dims+k]
                                                + outer*inner) ;
                                    }
                                }
                                mergedFlags[idx] = true ;
                            }
                        }
                    }
                }
                for ( int j = 0 ; j < dims*dims ; j++)
                {
                    sval0 = feature.cov[j] ;
                    sumByReduction(sdata,sval0,tid);
                    if ( tid == 0 )
                        mergedFeature.cov[j] = sdata[0]/mergedFeature.weight ;
                    __syncthreads() ;
                }

                if ( tid == 0 )
                {
//                    printf("saving merged feature\n") ;
                    force_symmetric_covariance(mergedFeature) ;
                    if (isPosDef(mergedFeature)){
                        int mergeIdx = update_offset + mergedSize ;
                        copy_gaussians(mergedFeature,mergedFeatures[mergeIdx]) ;
                        mergedSize++ ;
                    }
                    else
                    {
                        printf("discarding non PD feature (w = %f)\n",mergedFeature.weight) ;
                        // TODO: bad hack, just keep the max feature
                        int mergeIdx = update_offset + mergedSize ;
                        copy_gaussians(maxFeature,mergedFeatures[mergeIdx]) ;
                        mergedSize++ ;
                    }
                }
                __syncthreads() ;

            }
            __syncthreads() ;
            // save the merged map size
            if ( tid == 0 )
                mergedSizes[map_idx] = mergedSize ;
        }
    } // end loop over particles
    return ;
}

//__global__ void
//recombineFeaturesKernel(Gaussian6D* features_in, Gaussian6D* features_out,
//                        int* map_offsets_in, int* map_offsets_out, int n_maps,
//                        int n_features,
//                        Gaussian6D* features_combined, int* indices_combined)
//{
//    int tid = threadIdx.x + blockIdx.x*blockDim.x ;
//    if (tid < n_features){
//        int map_idx = 0 ;
//        bool out = false ;

//        int offset_in = 0 ;
//        int offset_out = 0 ;

//        int offset_old = 0 ;

//        int idx = 0 ;
//        for (int n = 0 ; n < n_maps ; n++ ){
//            offset_old = offset_out ;

//            offset_in = map_offsets_in[n+1] + map_offsets_out[n] ;
//            offset_out = map_offsets_in[n+1] + map_offsets_out[n+1] ;

//            if (offset_in >= tid){
//                indices_combined[tid] = n ;
//                out = false;
//                idx = (tid - offset_old) + map_offsets_in[n] ;
//                break ;
//            }
//            else if (offset_out >= tid){
//                indices_combined[tid] = n ;
//                out = true ;
//                idx = (tid - offset_in) + map_offsets_out[n] ;
//                break ;
//            }
//        }

//        if (out)
//            features_combined[tid] = features_out[idx] ;
//        else
//            features_combined[tid] = features_in[idx] ;

//    }
//}

/// given two vectors with defined segments, combine them in an
/// interleaved fashion
///
/// Example:
/// A = {a0,a1,a2,a3,a4,a5,a6,a7,a8,a9} ;
/// B = {b0,b1,b2,b3,b4}
/// offsets_vec1 = {0,3,5}
/// offsets_vec2 = {0,1,3}
///
/// Result:
/// combined = {a0,a1,a2,b0,a3,a4,b1,b2,a5,a6,a7,a8,a9,b3,b4}
/// indices_combined = {0,0,0,0,0,1,1,1,1,2,2,2,2,2,2,2}
///
/// Execution configuration: number of total threads should be greater
/// than or equal to total number of items.
template <typename T>
__global__ void
interleaveKernel(T* items1, T* items2, int* offsets_vec1, int* offsets_vec2,
                 int n_segments, int n_items,
                 T* combined, int* indices_combined)
{
    int tid0 = threadIdx.x + blockIdx.x*blockDim.x ;
    for ( int tid = tid0 ; tid < n_items ; tid+=gridDim.x*blockDim.x){
        int idx = 0 ;
        bool copy2 = false ;

        int offset1 = 0 ;
        int offset2 = 0 ;
        int offset_old = 0 ;

        for ( int n = 0 ; n < n_segments ; n++ ){
            offset_old = offset2 ;
            offset1 = offsets_vec1[n+1] + offsets_vec2[n] ;
            offset2 = offsets_vec1[n+1] + offsets_vec2[n+1] ;

            if ( offset1 > tid){
                indices_combined[tid] = n ;
                copy2 = false ;
                idx = (tid - offset_old) + offsets_vec1[n] ;
                break ;
            }
            else if (offset2 > tid){
                indices_combined[tid] = n ;
                copy2 = true ;
                idx = (tid - offset1) + offsets_vec2[n] ;
                break ;
            }
        }

        if (copy2)
            combined[tid] = items2[idx] ;
        else
            combined[tid] = items1[idx] ;
    }
}


/// expand values
/// example: values = {1,2,3,4} , factor = 3
///          expanded = {1,1,1,2,2,2,3,3,3,4,4,4} ;
/// launch config: total threads >= expanded count
__global__ void
expandKernel(double* values, int n_original, int factor, double* expanded){
    int tid0 = threadIdx.x + blockIdx.x*blockDim.x ;
    int stride = blockDim.x*gridDim.x ;
    for ( int tid = tid0 ; tid < n_original*factor ; tid += stride){
        int idx = floor(double(tid)/factor) ;
        expanded[tid] = values[idx] ;
    }
}

/// normalize the weights corresponding to each thread block
/// in place operation
/// launch configuration should be:
/// <<<num_features,particles_per_feature>>>
__global__ void
normalizeWeightsKernel(double* weights, int n_particles)
{
    __shared__ double shmem[256] ;
    int n_features = gridDim.x ;

    for ( int n = blockIdx.x ; n < n_features ; n+=gridDim.x ){
        double val = 0 ;
        for ( int i = threadIdx.x ; i < n_particles ; i+=blockDim.x ){
            int idx = i + n*n_particles ;
            val += weights[idx] ;
        }
        sumByReduction(shmem,val,threadIdx.x);
        __syncthreads() ;
        double sum = shmem[0] ;

        for ( int i = threadIdx.x ; i < n_particles ; i+=blockDim.x){
            int idx = i + n*n_particles ;
            double normed_weight = safeLog(weights[idx]) - safeLog(sum) ;
            weights[idx] = exp(normed_weight) ;
        }
    }
}

/// resample disparity space particles
/// launch configuration should be <<<num_features, particles_per_feature>>>
__global__ void
resampleFeaturesKernel(double* u, double* v, double* d,
                       double* vu, double* vv, double* vd,
                       double* weights, double* randvals, int n_features,
                       double* u_sampled, double* v_sampled, double* d_sampled,
                       double* vu_sampled, double* vv_sampled, double* vd_sampled)
{
    // each block corresponds to 1 feature. there may be more features
    // than the maximum number of blocks, so we use this for loop

    int n_particles = blockDim.x ;

    for ( int n = blockIdx.x ; n < n_features; n += gridDim.x ){
        double interval = 1.0/n_particles ;
        double r = randvals[n] + threadIdx.x*interval ;

        int offset = blockDim.x*n ;
        double c = weights[offset] ;
        int idx = offset ;
        while ( r > c ){
            c += weights[++idx] ;

            if (idx == offset + n_particles){
                idx-- ;
                break ;
            }
        }

        int idx_new = n*blockDim.x + threadIdx.x ;
        u_sampled[idx_new] = u[idx] ;
        v_sampled[idx_new] = v[idx] ;
        d_sampled[idx_new] = d[idx] ;
        vu_sampled[idx_new] = vu[idx] ;
        vv_sampled[idx_new] = vv[idx] ;
        vd_sampled[idx_new] = vd[idx] ;
    }
}

// template instantiation

template __global__ void
interleaveKernel(double*, double*, int*, int*, int, int, double*, int*) ;

template __global__ void
interleaveKernel(Gaussian6D*, Gaussian6D*, int*, int*, int, int, Gaussian6D*, int*) ;
