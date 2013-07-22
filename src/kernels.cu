#include "kernels.cuh"
#include "types.h"
#include "device_math.cuh"

__global__ void
fitGaussiansKernel(float* uArray, float* vArray, float* dArray,
                   float* vuArray, float* vvArray, float* vdArray,
                   float* weights,int nGaussians,
                   Gaussian6D* gaussians,
                   int n_particles){
    int tid = threadIdx.x ;
    __shared__ float sdata[256] ;
    for (int i = blockIdx.x ; i < nGaussians ; i+=gridDim.x){
        int offset = i*n_particles ;
        float val = 0 ;

        // compute mean u
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += uArray[offset+j] ;
        }
        sumByReduction(sdata,val,tid);
        float uMean = sdata[0]/n_particles ;
        __syncthreads() ;

        // compute mean v
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += vArray[offset+j] ;
        }
        sumByReduction(sdata,val,tid);
        float vMean = sdata[0]/n_particles ;
        __syncthreads() ;

        // compute mean d
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += dArray[offset+j] ;
        }
        sumByReduction(sdata,val,tid);
        float dMean = sdata[0]/n_particles ;
        __syncthreads() ;

        // compute mean vu
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += vuArray[offset+j] ;
        }
        sumByReduction(sdata,val,tid);
        float vuMean = sdata[0]/n_particles ;
        __syncthreads() ;

        // compute mean vv
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += vvArray[offset+j] ;
        }
        sumByReduction(sdata,val,tid);
        float vvMean = sdata[0]/n_particles ;
        __syncthreads() ;

        // compute mean vd
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += vdArray[offset+j] ;
        }
        sumByReduction(sdata,val,tid);
        float vdMean = sdata[0]/n_particles ;
        __syncthreads() ;


        // write means to output
        if (tid == 0){
//            cuPrintf("%f %f %f\n",uMean,vMean,dMean) ;
            gaussians[i].weight = weights[i] ;
            gaussians[i].mean[0] = uMean ;
            gaussians[i].mean[1] = vMean ;
            gaussians[i].mean[2] = dMean ;
            gaussians[i].mean[3] = vuMean ;
            gaussians[i].mean[4] = vvMean ;
            gaussians[i].mean[5] = vdMean ;
        }

        // covariance term u-u
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += pow(uArray[offset+j]-uMean,2) ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0)
            gaussians[i].cov[0] = sdata[0]/(n_particles-1) ;
        __syncthreads() ;

        // covariance term v-v
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += pow(vArray[offset+j]-vMean,2) ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0)
            gaussians[i].cov[7] = sdata[0]/(n_particles-1) ;
        __syncthreads() ;

        // covariance term d-d
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += pow(dArray[offset+j]-dMean,2) ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0)
            gaussians[i].cov[14] = sdata[0]/(n_particles-1) ;
        __syncthreads() ;

        // covariance term vu-vu
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += pow(vuArray[offset+j]-vuMean,2) ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0)
            gaussians[i].cov[21] = sdata[0]/(n_particles-1) ;
        __syncthreads() ;

        // covariance term vv-vv
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += pow(vvArray[offset+j]-vvMean,2) ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0)
            gaussians[i].cov[28] = sdata[0]/(n_particles-1) ;
        __syncthreads() ;

        // covariance term vd-vd
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += pow(vdArray[offset+j]-vdMean,2) ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0)
            gaussians[i].cov[35] = sdata[0]/(n_particles-1) ;
        __syncthreads() ;

        // covariance term u-v
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += (uArray[offset+j]-uMean)*(vArray[offset+j]-vMean) ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0){
            gaussians[i].cov[1] = sdata[0]/(n_particles-1) ;
            gaussians[i].cov[6] = gaussians[i].cov[1] ;
        }
        __syncthreads() ;

        // covariance term u-d
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += (uArray[offset+j]-uMean)*(dArray[offset+j]-dMean) ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0){
            gaussians[i].cov[2] = sdata[0]/(n_particles-1) ;
            gaussians[i].cov[12] = gaussians[i].cov[2] ;
        }
        __syncthreads() ;

        // covariance term u-vu
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += (uArray[offset+j]-uMean)*(vuArray[offset+j]-vuMean) ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0){
            gaussians[i].cov[3] = sdata[0]/(n_particles-1) ;
            gaussians[i].cov[18] = gaussians[i].cov[3] ;
        }
        __syncthreads() ;

        // covariance term u-vv
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += (uArray[offset+j]-uMean)*(vvArray[offset+j]-vvMean) ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0){
            gaussians[i].cov[4] = sdata[0]/(n_particles-1) ;
            gaussians[i].cov[24] = gaussians[i].cov[4] ;
        }
        __syncthreads() ;

        // covariance term u-vd
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += (uArray[offset+j]-uMean)*(vdArray[offset+j]-vdMean) ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0){
            gaussians[i].cov[5] = sdata[0]/(n_particles-1) ;
            gaussians[i].cov[30] = gaussians[i].cov[5] ;
        }
        __syncthreads() ;

        // covariance term v-d
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += (vArray[offset+j]-vMean)*(dArray[offset+j]-dMean) ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0){
            gaussians[i].cov[8] = sdata[0]/(n_particles-1) ;
            gaussians[i].cov[13] = gaussians[i].cov[8] ;
        }
        __syncthreads() ;

        // covariance term v-vu
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += (vArray[offset+j]-vMean)*(vuArray[offset+j]-vuMean) ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0){
            gaussians[i].cov[9] = sdata[0]/(n_particles-1) ;
            gaussians[i].cov[19] = gaussians[i].cov[9] ;
        }
        __syncthreads() ;

        // covariance term v-vv
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += (vArray[offset+j]-vMean)*(vvArray[offset+j]-vvMean) ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0){
            gaussians[i].cov[10] = sdata[0]/(n_particles-1) ;
            gaussians[i].cov[25] = gaussians[i].cov[10] ;
        }
        __syncthreads() ;

        // covariance term v-vd
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += (vArray[offset+j]-vMean)*(vdArray[offset+j]-vdMean) ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0){
            gaussians[i].cov[11] = sdata[0]/(n_particles-1) ;
            gaussians[i].cov[31] = gaussians[i].cov[11] ;
        }
        __syncthreads() ;

        // covariance term d-vu
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += (dArray[offset+j]-dMean)*(vuArray[offset+j]-vuMean) ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0){
            gaussians[i].cov[15] = sdata[0]/(n_particles-1) ;
            gaussians[i].cov[20] = gaussians[i].cov[15] ;
        }
        __syncthreads() ;

        // covariance term d-vv
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += (dArray[offset+j]-dMean)*(vvArray[offset+j]-vvMean) ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0){
            gaussians[i].cov[16] = sdata[0]/(n_particles-1) ;
            gaussians[i].cov[26] = gaussians[i].cov[16] ;
        }
        __syncthreads() ;

        // covariance term d-vd
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += (dArray[offset+j]-dMean)*(vdArray[offset+j]-vdMean) ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0){
            gaussians[i].cov[17] = sdata[0]/(n_particles-1) ;
            gaussians[i].cov[32] = gaussians[i].cov[17] ;
        }
        __syncthreads() ;

        // covariance term vu-vv
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += (vuArray[offset+j]-vuMean)*(vvArray[offset+j]-vvMean) ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0){
            gaussians[i].cov[22] = sdata[0]/(n_particles-1) ;
            gaussians[i].cov[27] = gaussians[i].cov[22] ;
        }
        __syncthreads() ;

        // covariance term vu-vd
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += (vuArray[offset+j]-vuMean)*(vdArray[offset+j]-vdMean) ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0){
            gaussians[i].cov[23] = sdata[0]/(n_particles-1) ;
            gaussians[i].cov[33] = gaussians[i].cov[23] ;
        }
        __syncthreads() ;

        // covariance term vv-vd
        val = 0 ;
        for(int j = tid ; j < n_particles ; j+=blockDim.x){
            val += (vvArray[offset+j]-vvMean)*(vdArray[offset+j]-vdMean) ;
        }
        sumByReduction(sdata,val,tid);
        if (tid == 0){
            gaussians[i].cov[29] = sdata[0]/(n_particles-1) ;
            gaussians[i].cov[34] = gaussians[i].cov[29] ;
        }
        __syncthreads() ;
    }
}

__global__ void
updateKernel(Gaussian6D* nondetections, Gaussian6D* detections,
             Gaussian6D* births, float* normalizers, int* map_offsets,
             int n_maps, int n_measure, Gaussian6D* updated,
             bool* merge_flags, float min_weight){
    int tid = threadIdx.x ;
    int idx_map = blockIdx.x ;

    int n_total_features = map_offsets[n_maps] ;
    int predict_offset = map_offsets[idx_map] ;
    int update_offset = map_offsets[idx_map]*(n_measure+1) +
            idx_map*n_measure;
    int n_features = map_offsets[idx_map+1] - map_offsets[idx_map] ;
    int n_update = n_features + n_features*n_measure + n_measure ;
    for ( int j = tid ; j < n_update ; j+= blockDim.x ){
        int idx_update = update_offset + j ;
        Gaussian6D feature ;
        if ( j < n_features ){
            int idx = predict_offset + j ;
            feature = nondetections[idx] ;
        }
        else if (j >= n_features && j < n_features*(n_measure+1)){
            int idx_measure = int(j-n_features)/n_features ;
            int idx_feature = fmod(float(j-n_features),n_features) ;
            int idx_detect = idx_measure*n_total_features + predict_offset + idx_feature ;
            feature = detections[idx_detect] ;

            int idx_normalizer = idx_measure*n_maps + idx_map ;
            feature.weight -= normalizers[idx_normalizer] ;
        }
        else if (j >= n_features*(n_measure+1) ){
            int idx = j - (n_features*(n_measure+1)) ;
            feature = births[idx] ;

            int idx_measure = idx ;
            int idx_normalizer = idx_measure*n_maps + idx_map ;
            feature.weight -= normalizers[idx_normalizer] ;
        }
        updated[idx_update] = feature ;

        if (feature.weight >= min_weight)
            merge_flags[idx_update] = true ;
        else
            merge_flags[idx_update] = false ;
    }
}

__global__ void
phdUpdateMergeKernel(Gaussian6D* updated_features,
                     Gaussian6D* mergedFeatures, int* mergedSizes,
                     bool* mergedFlags, int* map_offsets, int n_particles,
                     float min_separation)
{
    __shared__ Gaussian6D maxFeature ;
    __shared__ Gaussian6D mergedFeature ;
    __shared__ float sdata[256] ;
    __shared__ int mergedSize ;
    __shared__ int update_offset ;
    __shared__ int n_update ;
    int tid = threadIdx.x ;
    float dist ;
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
                else if(tid == 0)
                    maxFeature = updated_features[ (unsigned int)sdata[0] ] ;
                __syncthreads() ;

                // find features to merge with max feature
                float sval0 = 0 ;
//                float sval1 = 0 ;
//                float sval2 = 0 ;
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
                                    float outer = feature.mean[j] ;
                                    for ( int k = 0 ; k < dims ; k++)
                                    {
                                        float inner = feature.mean[k] ;
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
                    force_symmetric_covariance(mergedFeature) ;
                    int mergeIdx = update_offset + mergedSize ;
                    copy_gaussians(mergedFeature,mergedFeatures[mergeIdx]) ;
                    mergedSize++ ;
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
