#include "scphdcameracalibration.cuh"
#include "kernels.cuh"
#include "device_math.cuh"
#include "types.h"

#include <iostream>
#include <list>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <thrust/tuple.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/for_each.h>
#include <thrust/scan.h>
#include <thrust/random.h>
#include <thrust/random/normal_distribution.h>
#include <thrust/reduce.h>
#include <thrust/gather.h>
#include <thrust/fill.h>
#include <thrust/copy.h>

using namespace thrust ;
using namespace libconfig ;

typedef struct{
    int n ;
    DisparityPoint mean ;
    DisparityPoint variance ;
    float cov_u_v ;
    float cov_u_d ;
    float cov_u_vu ;
    float cov_u_vv ;
    float cov_u_vd ;
    float cov_v_d ;
    float cov_v_vu ;
    float cov_v_vv ;
    float cov_v_vd ;
    float cov_d_vu ;
    float cov_d_vv ;
    float cov_d_vd ;
    float cov_vu_vv ;
    float cov_vu_vd ;
    float cov_vv_vd ;

    void initialize(){
       DisparityPoint zeros = {0,0,0,0,0,0} ;
       mean = zeros ;
       variance = zeros ;
       n = 0 ;
       cov_u_v = 0.0 ;
       cov_u_d = 0.0 ;
       cov_u_vu = 0.0 ;
       cov_u_vv = 0.0 ;
       cov_u_vd = 0.0 ;
       cov_v_d = 0.0 ;
       cov_v_vu = 0.0 ;
       cov_v_vv = 0.0 ;
       cov_v_vd = 0.0 ;
       cov_d_vu = 0.0 ;
       cov_d_vv = 0.0 ;
       cov_d_vd = 0.0 ;
       cov_vu_vv = 0.0 ;
       cov_vu_vd = 0.0 ;
       cov_vv_vd = 0.0 ;
    }
} DisparityStats ;

struct generate_gaussian_noise : public thrust::unary_function<unsigned int, float>
{
    thrust::random::experimental::normal_distribution<float> dist ;

    generate_gaussian_noise(float mean = 0.0, float std = 1.0){
        dist = thrust::random::experimental::normal_distribution<float>(mean,std) ;
    }

    __device__ __host__ float
    operator()(unsigned int tid){
        thrust::default_random_engine rng ;
        rng.discard(tid);

        return dist(rng) ;
    }
} ;

struct predict_transform{
    const float dt_ ;
    predict_transform(float dt) : dt_(dt) {}

    template <typename Tuple>
    __device__ void
    operator()(Tuple t){
        float x = get<0>(t) ;
        float y = get<1>(t) ;
        float z = get<2>(t) ;
        float vx = get<3>(t) ;
        float vy = get<4>(t) ;
        float vz = get<5>(t) ;
        float ax = get<6>(t) ;
        float ay = get<7>(t) ;
        float az = get<8>(t) ;

        get<0>(t) = x + vx*dt_ + 0.5*ax*dt_*dt_ ;
        get<1>(t) = y + vy*dt_ + 0.5*ay*dt_*dt_ ;
        get<2>(t) = z + vz*dt_ + 0.5*az*dt_*dt_ ;
        get<3>(t) = vx + ax*dt_ ;
        get<4>(t) = vy + ay*dt_ ;
        get<5>(t) = vz + az*dt_ ;
    }
};

struct world_to_disparity_transform{
    DisparityMeasurementModel model_ ;
    Extrinsics* extrinsics ;

    world_to_disparity_transform(DisparityMeasurementModel model,
                                 Extrinsics* e) :
        model_(model), extrinsics(e) {}

    template <typename Tuple>
    __device__ __host__ tuple<float,float,float,float,float,float>
    operator()(Tuple t){
        EuclideanPoint p_world ;
        p_world.x = get<0>(t) ;
        p_world.y = get<1>(t) ;
        p_world.z = get<2>(t) ;
        p_world.vx = get<3>(t) ;
        p_world.vy = get<4>(t) ;
        p_world.vz = get<5>(t) ;
        int idx = get<6>(t) ;
        Extrinsics e = extrinsics[idx] ;
        DisparityPoint p_disparity = model_.computeMeasurement(p_world,e) ;

        return make_tuple(p_disparity.u,
                        p_disparity.v,
                        p_disparity.d,
                        p_disparity.vu,
                        p_disparity.vv,
                        p_disparity.vd) ;
    }
};

struct disparity_to_world_transform{
    DisparityMeasurementModel model_ ;
    Extrinsics* extrinsics ;

    disparity_to_world_transform(DisparityMeasurementModel model,
                                 Extrinsics* e) :
        model_(model), extrinsics(e) {}

    template <typename Tuple>
    __device__ __host__ tuple<float,float,float,float,float,float>
    operator()(Tuple t){
        DisparityPoint p_disparity ;
        p_disparity.u = get<0>(t) ;
        p_disparity.v = get<1>(t) ;
        p_disparity.d = get<2>(t) ;
        p_disparity.vu = get<3>(t) ;
        p_disparity.vv = get<4>(t) ;
        p_disparity.vd = get<5>(t) ;
        Extrinsics e = extrinsics[get<6>(t)] ;

        EuclideanPoint p_world = model_.invertMeasurement(p_disparity,e) ;
        return make_tuple(p_world.x,p_world.y, p_world.z,
                          p_world.vx,p_world.vy,p_world.vz) ;
    }
};

struct pack_disparity_stats :
    public thrust::unary_function<const DisparityPoint&, DisparityStats>
{
    __host__ __device__
    DisparityStats operator()(const DisparityPoint& p){
        DisparityStats result ;
        result.mean = p ;
        result.n = 1 ;
        return result ;
    }
} ;

//struct aggregate_disparity_stats :
//    public thrust::binary_function<const DisparityStats&, const DisparityStats&,
//                                   DisparityStats>
//{
//    __host__ __device__
//    DisparityStats operator()(const DisparityStats &x, const DisparityStats &y){
//        DisparityStats result ;
//        int n = x.n + y.n ;
//        DisparityPoint delta = x.mean - y.mean ;
//        result.n = n ;
//        result.mean = x.mean + delta*y.n/n ;

//        result.variance = x.variance + y.variance ;
//        result.variance += delta*delta*x.n*y.n/n ;

//        result.cov_u_v = x.cov_u_v + y.cov_u_v ;
//        result.cov_u_v += delta.u*delta.v*x.n*y.n/n ;

//        result.cov_u_d = x.cov_u_d + y.cov_u_d ;
//        result.cov_u_d += delta.u*delta.d*x.n*y.n/n ;

//        result.cov_u_vu = x.cov_u_vu + y.cov_u_vu ;
//        result.cov_u_vu += delta.u*delta.vu*x.n*y.n/n ;

//        result.cov_u_vv = x.cov_u_vv + y.cov_u_vv ;
//        result.cov_u_vv += delta.u*delta.vv*x.n*y.n/n ;

//        result.cov_u_vd = x.cov_u_vd + y.cov_u_vd ;
//        result.cov_u_vd += delta.u*delta.vd*x.n*y.n/n ;

//        result.cov_v_d = x.cov_v_d + y.cov_v_d ;
//        result.cov_v_d += delta.v*delta.d*x.n*y.n/n ;

//        result.cov_v_vu = x.cov_v_vu + y.cov_v_vu ;
//        result.cov_v_vu += delta.v*delta.vu*x.n*y.n/n ;

//        result.cov_v_vv = x.cov_v_vv + y.cov_v_vv ;
//        result.cov_v_vv += delta.v*delta.vv*x.n*y.n/n ;

//        result.cov_v_vd = x.cov_v_vd + y.cov_v_vd ;
//        result.cov_v_vd += delta.v*delta.vd*x.n*y.n/n ;

//        result.cov_d_vu = x.cov_d_vu + y.cov_d_vu ;
//        result.cov_d_vu += delta.d*delta.vu*x.n*y.n/n ;

//        result.cov_d_vv = x.cov_d_vv + y.cov_d_vv ;
//        result.cov_d_vv += delta.d*delta.vv*x.n*y.n/n ;

//        result.cov_d_vd = x.cov_d_vd + y.cov_d_vd ;
//        result.cov_d_vd += delta.d*delta.vd*x.n*y.n/n ;

//        result.cov_vu_vv = x.cov_vu_vv + y.cov_vu_vv ;
//        result.cov_vu_vv += delta.vu*delta.vv*x.n*y.n/n ;

//        result.cov_vu_vd = x.cov_vu_vd + y.cov_vu_vd ;
//        result.cov_vu_vd += delta.vu*delta.vd*x.n*y.n/n ;

//        result.cov_vv_vd = x.cov_vv_vd + y.cov_vv_vd ;
//        result.cov_vv_vd += delta.vv*delta.vd*x.n*y.n/n ;

//        return result ;
//    }
//};

struct update_components
{
    float* u_ ;
    float* v_ ;
    Gaussian6D* features_ ;
    DisparityMeasurementModel model_ ;
    update_components(Gaussian6D* features, float* u, float* v,
                           DisparityMeasurementModel model) :
        features_(features), u_(u), v_(v), model_(model) {}

    template <typename T>
    __device__ void
    operator()(T t){
        // compute measurement and feature indices from tid
        int idx_feature = get<0>(t) ;
        int idx_measure = get<1>(t) ;

        Gaussian6D f = features_[idx_feature] ;
        Gaussian6D f_update ;
        float* p = f.cov ;

        float var_u = model_.std_u()*model_.std_u() ;
        float var_v = model_.std_v()*model_.std_v() ;
        float pd = model_.pd() ;

        // innovation vector
        float innov[2] ;
        innov[0] = f.mean[0] - u_[idx_measure] ;
        innov[1] = f.mean[1] - v_[idx_measure] ;

        // Innovation covariance
        float sigma[4] ;
        sigma[0] = p[0] + var_u;
        sigma[1] = p[1];
        sigma[2] = p[6];
        sigma[3] = p[7] + var_v;

        // enforce symmetry
        sigma[1] = (sigma[1]+sigma[2])/2 ;
        sigma[2] = sigma[1] ;

        // inverse sigma
        float s[4] ;
        float det = sigma[0]*sigma[3] - sigma[1]*sigma[2] ;
        s[0] = sigma[3]/det ;
        s[1] = -sigma[1]/det ;
        s[2] = -sigma[2]/det ;
        s[3] = sigma[0]/det ;

        // measurement likelihood
        float dist = innov[0]*innov[0]*s[0] +
                innov[0]*innov[1]*(s[1] + s[2]) +
                innov[1]*innov[1]*s[3] ;
        f_update.weight = safeLog(pd)
                + safeLog(f.weight)
                - 0.5*dist
                - safeLog(2*M_PI)
                - 0.5*safeLog(det) ;

        // Kalman gain K = PH/S
        float K[12] ;
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
        f_update.cov[0] = (1 - K[0]) * (p[0] * (1 - K[0]) - p[6] * K[6]) - K[6] * (p[1] * (1 - K[0]) - p[7] * K[6]) + var_u * (int) pow((double) K[0], (double) 2) + var_v * (int) pow((double) K[6], (double) 2);
        f_update.cov[1] = -K[1] * (p[0] * (1 - K[0]) - p[6] * K[6]) + (1 - K[7]) * (p[1] * (1 - K[0]) - p[7] * K[6]) + K[0] * var_u * K[1] + K[6] * var_v * K[7];
        f_update.cov[2] = -K[2] * (p[0] * (1 - K[0]) - p[6] * K[6]) - K[8] * (p[1] * (1 - K[0]) - p[7] * K[6]) + p[2] * (1 - K[0]) - p[8] * K[6] + K[0] * var_u * K[2] + K[6] * var_v * K[8];
        f_update.cov[3] = -K[3] * (p[0] * (1 - K[0]) - p[6] * K[6]) - K[9] * (p[1] * (1 - K[0]) - p[7] * K[6]) + p[3] * (1 - K[0]) - p[9] * K[6] + K[0] * var_u * K[3] + K[6] * var_v * K[9];
        f_update.cov[4] = -K[4] * (p[0] * (1 - K[0]) - p[6] * K[6]) - K[10] * (p[1] * (1 - K[0]) - p[7] * K[6]) + p[4] * (1 - K[0]) - p[10] * K[6] + K[0] * var_u * K[4] + K[6] * var_v * K[10];
        f_update.cov[5] = -K[5] * (p[0] * (1 - K[0]) - p[6] * K[6]) - K[11] * (p[1] * (1 - K[0]) - p[7] * K[6]) + p[5] * (1 - K[0]) - p[11] * K[6] + K[0] * var_u * K[5] + K[6] * var_v * K[11];
        f_update.cov[6] = (1 - K[0]) * (-p[0] * K[1] + p[6] * (1 - K[7])) - K[6] * (-p[1] * K[1] + p[7] * (1 - K[7])) + K[0] * var_u * K[1] + K[6] * var_v * K[7];
        f_update.cov[7] = -K[1] * (-p[0] * K[1] + p[6] * (1 - K[7])) + (1 - K[7]) * (-p[1] * K[1] + p[7] * (1 - K[7])) + var_u * (int) pow((double) K[1], (double) 2) + var_v * (int) pow((double) K[7], (double) 2);
        f_update.cov[8] = -K[2] * (-p[0] * K[1] + p[6] * (1 - K[7])) - K[8] * (-p[1] * K[1] + p[7] * (1 - K[7])) - p[2] * K[1] + p[8] * (1 - K[7]) + K[1] * var_u * K[2] + K[7] * var_v * K[8];
        f_update.cov[9] = -K[3] * (-p[0] * K[1] + p[6] * (1 - K[7])) - K[9] * (-p[1] * K[1] + p[7] * (1 - K[7])) - p[3] * K[1] + p[9] * (1 - K[7]) + K[1] * var_u * K[3] + K[7] * var_v * K[9];
        f_update.cov[10] = -K[4] * (-p[0] * K[1] + p[6] * (1 - K[7])) - K[10] * (-p[1] * K[1] + p[7] * (1 - K[7])) - p[4] * K[1] + p[10] * (1 - K[7]) + K[1] * var_u * K[4] + K[7] * var_v * K[10];
        f_update.cov[11] = -K[5] * (-p[0] * K[1] + p[6] * (1 - K[7])) - K[11] * (-p[1] * K[1] + p[7] * (1 - K[7])) - p[5] * K[1] + p[11] * (1 - K[7]) + K[1] * var_u * K[5] + K[7] * var_v * K[11];
        f_update.cov[12] = (1 - K[0]) * (-p[0] * K[2] - p[6] * K[8] + p[12]) - K[6] * (-p[1] * K[2] - p[7] * K[8] + p[13]) + K[0] * var_u * K[2] + K[6] * var_v * K[8];
        f_update.cov[13] = -K[1] * (-p[0] * K[2] - p[6] * K[8] + p[12]) + (1 - K[7]) * (-p[1] * K[2] - p[7] * K[8] + p[13]) + K[1] * var_u * K[2] + K[7] * var_v * K[8];
        f_update.cov[14] = -K[2] * (-p[0] * K[2] - p[6] * K[8] + p[12]) - K[8] * (-p[1] * K[2] - p[7] * K[8] + p[13]) - p[2] * K[2] - p[8] * K[8] + p[14] + var_u * (int) pow((double) K[2], (double) 2) + var_v * (int) pow((double) K[8], (double) 2);
        f_update.cov[15] = -K[3] * (-p[0] * K[2] - p[6] * K[8] + p[12]) - K[9] * (-p[1] * K[2] - p[7] * K[8] + p[13]) - p[3] * K[2] - p[9] * K[8] + p[15] + K[2] * var_u * K[3] + K[8] * var_v * K[9];
        f_update.cov[16] = -K[4] * (-p[0] * K[2] - p[6] * K[8] + p[12]) - K[10] * (-p[1] * K[2] - p[7] * K[8] + p[13]) - p[4] * K[2] - p[10] * K[8] + p[16] + K[2] * var_u * K[4] + K[8] * var_v * K[10];
        f_update.cov[17] = -K[5] * (-p[0] * K[2] - p[6] * K[8] + p[12]) - K[11] * (-p[1] * K[2] - p[7] * K[8] + p[13]) - p[5] * K[2] - p[11] * K[8] + p[17] + K[2] * var_u * K[5] + K[8] * var_v * K[11];
        f_update.cov[18] = (1 - K[0]) * (-p[0] * K[3] - p[6] * K[9] + p[18]) - K[6] * (-p[1] * K[3] - p[7] * K[9] + p[19]) + K[0] * var_u * K[3] + K[6] * var_v * K[9];
        f_update.cov[19] = -K[1] * (-p[0] * K[3] - p[6] * K[9] + p[18]) + (1 - K[7]) * (-p[1] * K[3] - p[7] * K[9] + p[19]) + K[1] * var_u * K[3] + K[7] * var_v * K[9];
        f_update.cov[20] = -K[2] * (-p[0] * K[3] - p[6] * K[9] + p[18]) - K[8] * (-p[1] * K[3] - p[7] * K[9] + p[19]) - p[2] * K[3] - p[8] * K[9] + p[20] + K[2] * var_u * K[3] + K[8] * var_v * K[9];
        f_update.cov[21] = -K[3] * (-p[0] * K[3] - p[6] * K[9] + p[18]) - K[9] * (-p[1] * K[3] - p[7] * K[9] + p[19]) - p[3] * K[3] - p[9] * K[9] + p[21] + var_u * (int) pow((double) K[3], (double) 2) + var_v * (int) pow((double) K[9], (double) 2);
        f_update.cov[22] = -K[4] * (-p[0] * K[3] - p[6] * K[9] + p[18]) - K[10] * (-p[1] * K[3] - p[7] * K[9] + p[19]) - p[4] * K[3] - p[10] * K[9] + p[22] + K[3] * var_u * K[4] + K[9] * var_v * K[10];
        f_update.cov[23] = -K[5] * (-p[0] * K[3] - p[6] * K[9] + p[18]) - K[11] * (-p[1] * K[3] - p[7] * K[9] + p[19]) - p[5] * K[3] - p[11] * K[9] + p[23] + K[3] * var_u * K[5] + K[9] * var_v * K[11];
        f_update.cov[24] = (1 - K[0]) * (-p[0] * K[4] - p[6] * K[10] + p[24]) - K[6] * (-p[1] * K[4] - p[7] * K[10] + p[25]) + K[0] * var_u * K[4] + K[6] * var_v * K[10];
        f_update.cov[25] = -K[1] * (-p[0] * K[4] - p[6] * K[10] + p[24]) + (1 - K[7]) * (-p[1] * K[4] - p[7] * K[10] + p[25]) + K[1] * var_u * K[4] + K[7] * var_v * K[10];
        f_update.cov[26] = -K[2] * (-p[0] * K[4] - p[6] * K[10] + p[24]) - K[8] * (-p[1] * K[4] - p[7] * K[10] + p[25]) - p[2] * K[4] - p[8] * K[10] + p[26] + K[2] * var_u * K[4] + K[8] * var_v * K[10];
        f_update.cov[27] = -K[3] * (-p[0] * K[4] - p[6] * K[10] + p[24]) - K[9] * (-p[1] * K[4] - p[7] * K[10] + p[25]) - p[3] * K[4] - p[9] * K[10] + p[27] + K[3] * var_u * K[4] + K[9] * var_v * K[10];
        f_update.cov[28] = -K[4] * (-p[0] * K[4] - p[6] * K[10] + p[24]) - K[10] * (-p[1] * K[4] - p[7] * K[10] + p[25]) - p[4] * K[4] - p[10] * K[10] + p[28] + var_u * (int) pow((double) K[4], (double) 2) + var_v * (int) pow((double) K[10], (double) 2);
        f_update.cov[29] = -K[5] * (-p[0] * K[4] - p[6] * K[10] + p[24]) - K[11] * (-p[1] * K[4] - p[7] * K[10] + p[25]) - p[5] * K[4] - p[11] * K[10] + p[29] + K[4] * var_u * K[5] + K[10] * var_v * K[11];
        f_update.cov[30] = (1 - K[0]) * (-p[0] * K[5] - p[6] * K[11] + p[30]) - K[6] * (-p[1] * K[5] - p[7] * K[11] + p[31]) + K[0] * var_u * K[5] + K[6] * var_v * K[11];
        f_update.cov[31] = -K[1] * (-p[0] * K[5] - p[6] * K[11] + p[30]) + (1 - K[7]) * (-p[1] * K[5] - p[7] * K[11] + p[31]) + K[1] * var_u * K[5] + K[7] * var_v * K[11];
        f_update.cov[32] = -K[2] * (-p[0] * K[5] - p[6] * K[11] + p[30]) - K[8] * (-p[1] * K[5] - p[7] * K[11] + p[31]) - p[2] * K[5] - p[8] * K[11] + p[32] + K[2] * var_u * K[5] + K[8] * var_v * K[11];
        f_update.cov[33] = -K[3] * (-p[0] * K[5] - p[6] * K[11] + p[30]) - K[9] * (-p[1] * K[5] - p[7] * K[11] + p[31]) - p[3] * K[5] - p[9] * K[11] + p[33] + K[3] * var_u * K[5] + K[9] * var_v * K[11];
        f_update.cov[34] = -K[4] * (-p[0] * K[5] - p[6] * K[11] + p[30]) - K[10] * (-p[1] * K[5] - p[7] * K[11] + p[31]) - p[4] * K[5] - p[10] * K[11] + p[34] + K[4] * var_u * K[5] + K[10] * var_v * K[11];
        f_update.cov[35] = -K[5] * (-p[0] * K[5] - p[6] * K[11] + p[30]) - K[11] * (-p[1] * K[5] - p[7] * K[11] + p[31]) - p[5] * K[5] - p[11] * K[11] + p[35] + var_u * (int) pow((double) K[5], (double) 2) + var_v * (int) pow((double) K[11], (double) 2);

        get<2>(t) = f_update ;
    }
};

struct compute_birth{
    const DisparityPoint birth_means ;
    const DisparityPoint birth_vars ;

    compute_birth(DisparityPoint means0, DisparityPoint vars0) :
        birth_means(means0), birth_vars(vars0)
    {}

    template<typename T>
    __device__ void
    operator()(T t){
        float u = get<0>(t) ;
        float v = get<1>(t) ;
        Gaussian6D feature_birth ;
        feature_birth.mean[0] = u ;
        feature_birth.mean[1] = v ;
        feature_birth.mean[2] = birth_means.d ;
        feature_birth.mean[3] = birth_means.vu ;
        feature_birth.mean[4] = birth_means.vv ;
        feature_birth.mean[5] = birth_means.vd ;

        for ( int i = 0 ; i < 36 ; i++)
            feature_birth.cov[i] = 0 ;
        feature_birth.cov[0] = birth_vars.u ;
        feature_birth.cov[7] = birth_vars.v ;
        feature_birth.cov[14] = birth_vars.d ;
        feature_birth.cov[21] = birth_vars.vu ;
        feature_birth.cov[28] = birth_vars.vv ;
        feature_birth.cov[35] = birth_vars.vd ;

        get<2>(t) = feature_birth ;
    }
};

struct compute_nondetect{
    const float pd_ ;
    compute_nondetect(float pd) : pd_(pd) {}

    __device__ Gaussian6D
    operator()(Gaussian6D feature_predict){
        Gaussian6D feature_nondetect = feature_predict ;
        feature_nondetect.weight += safeLog(pd_) ;
        return feature_nondetect ;
    }
};


struct log_sum_exp : public thrust::binary_function<const float, const float, float>
{
    __device__ __host__ float
    operator()(const float a, const float b){
        if (a > b){
            return a + safeLog(exp(b-a)) ;
        }
        else{
            return b + safeLog(exp(a-b)) ;
        }
    }
} ;

struct get_weight : public thrust::unary_function<const Gaussian6D, float>
{
    __device__ __host__ float
    operator()(const Gaussian6D g){
        return g.weight ;
    }
} ;

struct times : public thrust::unary_function<const int, int>
{
    const int C_ ;
    times(int c) : C_(c) {}

    __device__ __host__ int
    operator()(const int x){ return C_*x ; }
} ;

template <typename T1, typename T2>
struct divide_by : public thrust::unary_function<T1,T2>
{
    const T2 C_ ;
    divide_by(T2 c) : C_(c) {}

    __device__ __host__ T2
    operator()(const T1 x){ return x/C_ ; }
} ;

template <typename T1, typename T2>
struct exponentiate : public thrust::unary_function<T1,T2>
{
    __device__ __host__ T2
    operator()(T1 x){ return exp(x) ; }
} ;


struct sample_disparity_gaussian
{
    const int n_samples_ ;
    const Gaussian6D* gaussians_ ;
    sample_disparity_gaussian(Gaussian6D* g, int n) : n_samples_(n), gaussians_(g) {}

    // number of parallel threads is n_features*samples_per_feature
    // tuple argument is (tid, u, v, d, vu, vv, vd) ;
    template<typename T>
    __device__ void
    operator()(T t){
        int tid = get<0>(t) ;

        // generate uncorrelated normally distributed values
        thrust::default_random_engine rng ;
        rng.discard(tid) ;
        thrust::random::experimental::normal_distribution<float> randn(0.0,1.0) ;
        float vals[6] ;
        for ( int i = 0 ; i < 6 ; i++)
            vals[i] = randn(rng) ;

        // uncorrelated values are transformed by multiplying with cholesky
        // decomposition of covariance matrix, and adding mean
        int idx = int(tid/n_samples_) ;
        Gaussian6D feature = gaussians_[idx] ;

        float L[36] ;
        cholesky(feature.cov,L,6);

        get<1>(t) = L[0]*vals[0] + feature.mean[0] ;
        get<2>(t) = L[1]*vals[0] + L[7]*vals[1] + feature.mean[1] ;
        get<3>(t) = L[2]*vals[0] + L[8]*vals[1] + L[14]*vals[2]
                + feature.mean[2] ;
        get<4>(t) = L[3]*vals[0] + L[9]*vals[1] + L[15]*vals[2]
                + L[21]*vals[3] + feature.mean[3] ;
        get<5>(t) = L[4]*vals[0] + L[10]*vals[1] + L[16]*vals[2]
                + L[22]*vals[3] + L[28]*vals[4] + feature.mean[4] ;
        get<6>(t) = L[5]*vals[0] + L[11]*vals[1] + L[17]*vals[2]
                + L[23]*vals[3] + L[29]*vals[4] + L[35]*vals[5]
                + feature.mean[5] ;
    }
};

SCPHDCameraCalibration::SCPHDCameraCalibration(const char* config_file)
{
    config_.readFile(config_file);

    cout << "configuring particle counts" << endl ;
    n_particles_ = config_.lookup("n_particles") ;
    particles_per_feature_ = config_.lookup("particles_per_feature") ;

    // initialize calibration prior
    cout << "configuring prior distribution" << endl ;
    const Setting& prior_cartesian = config_.lookup("prior.cartesian") ;
    const Setting& prior_angular = config_.lookup("prior.angular") ;

    EuclideanPoint prior_mean_cartesian ;
    prior_mean_cartesian.x = prior_cartesian["x"]["mean"] ;
    prior_mean_cartesian.y = prior_cartesian["y"]["mean"] ;
    prior_mean_cartesian.z = prior_cartesian["z"]["mean"] ;
    prior_mean_cartesian.vx = prior_cartesian["vx"]["mean"] ;
    prior_mean_cartesian.vy = prior_cartesian["vy"]["mean"] ;
    prior_mean_cartesian.vz = prior_cartesian["vz"]["mean"] ;

    EuclideanPoint prior_std_cartesian ;
    prior_std_cartesian.x = prior_cartesian["x"]["std"] ;
    prior_std_cartesian.y = prior_cartesian["y"]["std"] ;
    prior_std_cartesian.z = prior_cartesian["z"]["std"] ;
    prior_std_cartesian.vx = prior_cartesian["vx"]["std"] ;
    prior_std_cartesian.vy = prior_cartesian["vy"]["std"] ;
    prior_std_cartesian.vz = prior_cartesian["vz"]["std"] ;

    EuclideanPoint prior_mean_angular ;
    prior_mean_angular.x = prior_angular["x"]["mean"] ;
    prior_mean_angular.y = prior_angular["y"]["mean"] ;
    prior_mean_angular.z = prior_angular["z"]["mean"] ;
    prior_mean_angular.vx = prior_angular["vx"]["mean"] ;
    prior_mean_angular.vy = prior_angular["vy"]["mean"] ;
    prior_mean_angular.vz = prior_angular["vz"]["mean"] ;

    EuclideanPoint prior_std_angular ;
    prior_std_angular.x = prior_angular["x"]["std"] ;
    prior_std_angular.y = prior_angular["y"]["std"] ;
    prior_std_angular.z = prior_angular["z"]["std"] ;
    prior_std_angular.vx = prior_angular["vx"]["std"] ;
    prior_std_angular.vy = prior_angular["vy"]["std"] ;
    prior_std_angular.vz = prior_angular["vz"]["std"] ;

    host_vector<Extrinsics> calibration_prior(n_particles_) ;

    thrust::default_random_engine rng ;
    thrust::random::experimental::normal_distribution<float> randn(0.0,1.0) ;
    for ( int n = 0 ; n < n_particles_ ; n++ ){
        calibration_prior[n].cartesian.x = randn(rng)*prior_std_cartesian.x
                + prior_mean_cartesian.x;
        calibration_prior[n].cartesian.y = randn(rng)*prior_std_cartesian.y
                + prior_mean_cartesian.y ;
        calibration_prior[n].cartesian.z = randn(rng)*prior_std_cartesian.z
                + prior_mean_cartesian.z;
        calibration_prior[n].cartesian.vx = randn(rng)*prior_std_cartesian.vx
                + prior_mean_cartesian.vx;
        calibration_prior[n].cartesian.vy = randn(rng)*prior_std_cartesian.vy
                + prior_mean_cartesian.vy;
        calibration_prior[n].cartesian.vz = randn(rng)*prior_std_cartesian.vz
                + prior_mean_cartesian.vz;

        calibration_prior[n].angular.x = randn(rng)*prior_std_angular.x
                + prior_mean_angular.x;
        calibration_prior[n].angular.y = randn(rng)*prior_std_angular.y
                + prior_mean_angular.y ;
        calibration_prior[n].angular.z = randn(rng)*prior_std_angular.z
                + prior_mean_angular.z;
        calibration_prior[n].angular.vx = randn(rng)*prior_std_angular.vx
                + prior_mean_angular.vx;
        calibration_prior[n].angular.vy = randn(rng)*prior_std_angular.vy
                + prior_mean_angular.vy;
        calibration_prior[n].angular.vz = randn(rng)*prior_std_angular.vz
                + prior_mean_angular.vz;
    }
    dev_particle_states_ = calibration_prior ;

    // create motion and measurement models
    cout << "configuring models" << endl ;
    const Setting& feature_motion_config = config_.lookup("feature_motion_model") ;
    feature_motion_model_ = LinearCVMotionModel3D(feature_motion_config["ax"],
                                                  feature_motion_config["ay"],
                                                  feature_motion_config["az"]) ;

    const Setting& measurement_config = config_.lookup("measurement_model") ;
    measurement_model_ = DisparityMeasurementModel(
            measurement_config["fx"], measurement_config["fy"],
            measurement_config["u0"],measurement_config["v0"],
            measurement_config["std_u"],measurement_config["std_v"],
            measurement_config["pd"],measurement_config["lambda"]) ;

}\

void SCPHDCameraCalibration::initializeCuda(){
    int n_cuda_devices = 0 ;
    checkCudaErrors( cudaGetDeviceCount(&n_cuda_devices) ) ;
    std::cout << "Found " << n_cuda_devices << "CUDA devices" << std::endl ;

    // select the device with the most multiprocesors
    int n_processors = -1 ;
    int best_device = -1 ;
    for ( int n = 0 ; n < n_cuda_devices ; n++){
        int tmp ;
        checkCudaErrors( cudaDeviceGetAttribute( &tmp, cudaDevAttrMultiProcessorCount, n) ) ;
        if (tmp > n_processors){
            best_device = n ;
            n_processors = tmp ;
        }
    }
    checkCudaErrors( cudaSetDevice(best_device) ) ;
    checkCudaErrors( cudaGetDeviceProperties(&cuda_dev_props_,best_device) ) ;
}

void SCPHDCameraCalibration::predict(float dt)
{
    // predict calibration

    // predict features
    int n_particles = dev_x_.size() ;
    device_vector<float> ax(n_particles) ;
    device_vector<float> ay(n_particles) ;
    device_vector<float> az(n_particles) ;

    thrust::transform(counting_iterator<int>(0), counting_iterator<int>(n_particles),
                      ax.begin(),generate_gaussian_noise(0,feature_motion_model_.std_ax())) ;
    thrust::transform(counting_iterator<int>(0), counting_iterator<int>(n_particles),
                      ay.begin(),generate_gaussian_noise(0,feature_motion_model_.std_ay())) ;
    thrust::transform(counting_iterator<int>(0), counting_iterator<int>(n_particles),
                      az.begin(),generate_gaussian_noise(0,feature_motion_model_.std_az())) ;

    thrust::for_each(make_zip_iterator(make_tuple(dev_x_.begin(),
                                                  dev_y_.begin(),
                                                  dev_z_.begin(),
                                                  dev_vx_.begin(),
                                                  dev_vy_.begin(),
                                                  dev_vz_.begin(),
                                                  ax.begin(),
                                                  ay.begin(),
                                                  az.begin())),
                     make_zip_iterator(make_tuple(dev_x_.end(),
                                                  dev_y_.end(),
                                                  dev_z_.end(),
                                                  dev_vx_.end(),
                                                  dev_vy_.end(),
                                                  dev_vz_.end(),
                                                  ax.end(),
                                                  ay.end(),
                                                  az.end())),
                     predict_transform(dt)) ;
}

void SCPHDCameraCalibration::computeDisparityParticles(){
    // make sure there is enough space to store disparity particles
    int n_particles = dev_x_.size() ;
    dev_u_.resize(n_particles);
    dev_v_.resize(n_particles);
    dev_d_.resize(n_particles);
    dev_vu_.resize(n_particles);
    dev_vv_.resize(n_particles);
    dev_vd_.resize(n_particles);

    
    thrust::transform(make_zip_iterator(make_tuple(dev_x_.begin(),
                                                   dev_y_.begin(),
                                                   dev_z_.begin(),
                                                   dev_vx_.begin(),
                                                   dev_vy_.begin(),
                                                   dev_vz_.begin(),
                                                   dev_particle_indices_.begin())),
                      make_zip_iterator(make_tuple(dev_x_.end(),
                                                   dev_y_.end(),
                                                   dev_z_.end(),
                                                   dev_vx_.end(),
                                                   dev_vy_.end(),
                                                   dev_vx_.end(),
                                                   dev_particle_indices_.end())),
                      make_zip_iterator(make_tuple(dev_u_.begin(),
                                                   dev_v_.begin(),
                                                   dev_d_.begin(),
                                                   dev_vu_.begin(),
                                                   dev_vv_.begin(),
                                                   dev_vd_.begin())),
                      world_to_disparity_transform(measurement_model_,
                                                   raw_pointer_cast(&dev_particle_states_[0]))
            );

}

void SCPHDCameraCalibration::computeEuclideanParticles(){
    // make sure there is enough space to store the particles
    int n_particles = dev_u_.size() ;
    dev_x_.resize(n_particles);
    dev_y_.resize(n_particles);
    dev_z_.resize(n_particles);
    dev_vx_.resize(n_particles);
    dev_vy_.resize(n_particles);
    dev_vz_.resize(n_particles);


    thrust::transform(make_zip_iterator(make_tuple(dev_u_.begin(),
                                                   dev_v_.begin(),
                                                   dev_d_.begin(),
                                                   dev_vv_.begin(),
                                                   dev_vv_.begin(),
                                                   dev_vd_.begin(),
                                                   dev_particle_indices_.begin())),
                      make_zip_iterator(make_tuple(dev_u_.end(),
                                                   dev_v_.end(),
                                                   dev_d_.end(),
                                                   dev_vu_.end(),
                                                   dev_vv_.end(),
                                                   dev_vd_.end(),
                                                   dev_particle_indices_.end())),
                      make_zip_iterator(make_tuple(dev_u_.begin(),
                                                   dev_v_.begin(),
                                                   dev_d_.begin(),
                                                   dev_vu_.begin(),
                                                   dev_vv_.begin(),
                                                   dev_vd_.begin())),
                      disparity_to_world_transform(measurement_model_,
                                                   raw_pointer_cast(&dev_particle_states_[0]))
            );
}

void SCPHDCameraCalibration::update(vector<float> u, vector<float> v,
                                    bool fixed_camera = false){
    int n_measure = u.size() ;
    int n_features = dev_features_.size() ;
    int n_detect = n_measure*n_features ;

    // transform euclidean particles to disparity space
    computeDisparityParticles() ;

    // fit gaussians
    fitGaussians() ;

    // compute map sizes
    device_vector<int> dev_map_sizes(n_particles_) ;
    thrust::reduce_by_key(dev_gaussian_indices_.begin(),
                          dev_gaussian_indices_.end(),
                          make_constant_iterator(1),
                          make_discard_iterator(),
                          dev_map_sizes.begin()) ;
    host_vector<int> map_sizes = dev_map_sizes ;

    // generate pre-update indices
    device_vector<int> idx_features(n_detect) ;
    device_vector<int> idx_measure(n_detect) ;
    device_vector<int>::iterator it_features = idx_features.begin() ;
    device_vector<int>::iterator it_measure = idx_measure.begin() ;
    for ( int n = 0 ; n < n_particles_ ; n++ ){
        for ( int m = 0 ; m < n_measure ; m++ ){
            it_measure = thrust::fill_n(it_measure,map_sizes[n],m) ;
            it_features = thrust::copy(make_counting_iterator(0),
                                       make_counting_iterator(map_sizes[n]),
                                       it_features) ;
        }
    }

//    for ( int m = 0 ; m < n_measure ; m++ ){
//        thrust::fill_n(idx_measure.begin()+offset,
//                       n_features,m) ;
//        thrust::copy(make_counting_iterator(0),
//                     make_counting_iterator(n_features),
//                     idx_features.begin()+offset) ;
//        offset += n_features ;
//    }
    thrust::device_vector<int> dev_idx_features = idx_features ;
    thrust::device_vector<int> dev_idx_measure = idx_measure ;

    // allocate space for update components
//    dev_features_preupdate_.resize(n_detect);
    device_vector<Gaussian6D> dev_features_detect(n_detect) ;

    // copy measurements to device
    device_vector<float> dev_measure_u = u ;
    device_vector<float> dev_measure_v = v ;

    // compute birth terms
    DisparityPoint birth_mean ;
    birth_mean.d = config_.lookup("d0") ;
    birth_mean.vu = config_.lookup("vu0") ;
    birth_mean.vv = config_.lookup("vv0") ;
    birth_mean.vd = config_.lookup("vd0") ;

    DisparityPoint birth_vars ;
    birth_vars.u = pow(measurement_model_.std_u(),2) ;
    birth_vars.v = pow(measurement_model_.std_v(),2) ;
    birth_vars.d = config_.lookup("var_d0") ;
    birth_vars.vu = config_.lookup("var_vu0") ;
    birth_vars.vv = config_.lookup("var_vv0") ;
    birth_vars.vd = config_.lookup("var_vd0") ;

    device_vector<Gaussian6D> dev_features_birth(n_measure) ;
    thrust::for_each(make_zip_iterator(make_tuple(
                        dev_measure_u.begin(),
                        dev_measure_v.begin(),
                        dev_features_birth.begin())),
                     make_zip_iterator(make_tuple(
                       dev_measure_u.end(),
                       dev_measure_v.end(),
                       dev_features_birth.end())),
                     compute_birth(birth_mean,birth_vars)) ;

//    // duplicate birth terms for each map
//    device_vector<Gaussian6D> dev_features_birth(n_measure*n_particles_) ;
//    device_vector<Gaussian6D>::iterator dest = dev_features_birth.begin() ;
//    for ( int i = 0 ; i < n_particles_ ; i++ ){
//        dest = thrust::copy_n(dev_features_birth_single.begin(),
//                              n_measure, dest) ;
//    }


    // compute non-detection terms
    device_vector<Gaussian6D> dev_features_nondetect(n_features) ;
    thrust::transform(dev_features_.begin(),dev_features_.end(),
                      dev_features_nondetect.begin(),
                      compute_nondetect(measurement_model_.pd())) ;

    // compute detection terms
    thrust::for_each(make_zip_iterator(make_tuple(
                        dev_idx_features.begin(),
                        dev_idx_measure.begin(),
                        dev_features_detect.begin())),
                     make_zip_iterator(make_tuple(
                        dev_idx_features.end(),
                        dev_idx_measure.end(),
                        dev_features_detect.end())),
                     update_components(
                        raw_pointer_cast(&dev_features_[0]),
                        raw_pointer_cast(&dev_measure_u[0]),
                        raw_pointer_cast(&dev_measure_v[0]),
                        measurement_model_)
                     ) ;

    // compute normalization terms
    // use thrust::reduce_by_key to sum only over terms from the same map,
    // and the same measurement

    // extract feature weights
    device_vector<float> dev_weights_detect(n_detect) ;
    thrust::transform(dev_features_detect.begin(),dev_features_detect.end(),
                      dev_weights_detect.begin(),get_weight()) ;


    // consecutive keys are features from the same measurement

    int n_normalizers = n_particles_*n_measure ;
    device_vector<float> dev_normalizers(n_normalizers) ;
    thrust::reduce_by_key(dev_idx_measure.begin(),
                          dev_idx_measure.end(),
                          dev_weights_detect.begin(),
                          make_discard_iterator(),
                          dev_normalizers.begin(),
                          log_sum_exp()) ;



    // add the clutter and birth intensities to the normalizers
    float birth_plus_clutter = safeLog(float(config_.lookup("w0")) +
                                       measurement_model_.kappa()) ;

    thrust::transform(dev_normalizers.begin(),
                      dev_normalizers.end(),
                      thrust::constant_iterator<float>(birth_plus_clutter),
                      dev_normalizers.begin(),
                      log_sum_exp()) ;

    // ------- SCPHD update ---------------------------------------------- //

    // allocate space for updated features
    int n_update = n_features + n_features*n_measure + n_measure*n_particles_ ;
    device_vector<Gaussian6D> dev_features_update(n_update);

    // compute the indexing offsets with an inclusive scan
    host_vector<int> map_offsets(n_particles_+ 1) ;
    map_offsets[0] = 0 ;
    thrust::inclusive_scan(map_sizes.begin(),map_sizes.end(),
                           map_offsets.begin()+1) ;
    device_vector<int> dev_map_offsets = map_offsets ;

    // create vector of flags to control GM merging
    device_vector<bool> dev_merge_flags(n_update) ;

    // lauch kernel for performing update
    updateKernel<<<n_particles_,256>>>(
            raw_pointer_cast(&dev_features_nondetect[0]),
            raw_pointer_cast(&dev_features_detect[0]),
            raw_pointer_cast(&dev_features_birth[0]),
            raw_pointer_cast(&dev_normalizers[0]),
            raw_pointer_cast(&dev_map_offsets[0]),
            n_particles_, n_measure,
            raw_pointer_cast(&dev_features_update[0]),
            raw_pointer_cast(&dev_merge_flags[0]),
            config_.lookup("min_weight"));



    // update parent particle weights
    if (!fixed_camera){
        // create key vector for reducing normalizers
        device_vector<int> dev_keys(n_normalizers) ;
        device_vector<int>::iterator it = dev_keys.begin() ;
        for ( int n = 0 ; n < n_particles_ ; n++ ){
            it = thrust::fill_n(it,n_measure,n) ;
        }

        // sum the log-valued normalizers
        device_vector<float> dev_normalizer_sums(n_particles_) ;
        thrust::reduce_by_key(dev_keys.begin(),dev_keys.end(),
                              dev_normalizers.begin(),
                              make_discard_iterator(),
                              dev_normalizer_sums.begin()) ;

        // compute predicted cardinalities
        device_vector<float> dev_cardinalities_predict(n_particles_) ;
        thrust::reduce_by_key(dev_gaussian_indices_.begin(),
                              dev_gaussian_indices_.end(),
                              dev_feature_weights_.begin(),
                              make_discard_iterator(),
                              dev_cardinalities_predict.begin()) ;

        // add predicted cardinalities to normalizer sums
        thrust::transform(dev_normalizer_sums.begin(),
                          dev_normalizer_sums.end(),
                          dev_cardinalities_predict.begin(),
                          dev_normalizer_sums.begin(),
                          thrust::plus<float>()) ;

        // exponentiate
        thrust::for_each(dev_normalizer_sums.begin(),
                         dev_normalizer_sums.end(),
                         exponentiate<float,float>()) ;

        // copy to host and reweight particles
        host_vector<float> normalizer_sums = dev_normalizer_sums ;

        thrust::transform(particle_weights_.begin(),
                          particle_weights_.end(),
                          normalizer_sums.begin(),
                          particle_weights_.begin(),
                          thrust::plus<float>()) ;

        // normalize particle weights
        float sum = thrust::reduce(particle_weights_.begin(),
                                 particle_weights_.end()) ;
        thrust::for_each(particle_weights_.begin(),
                         particle_weights_.end(),
                         divide_by<float,float>(sum)) ;
    }

    // ---------------- GM reduction ------------------------------------- //
    device_vector<int> dev_merged_sizes(n_particles_) ;
    device_vector<Gaussian6D> dev_gaussians_merged_tmp(n_update) ;

    // recalculate offsets for updated map size
    for ( int n = 0 ; n < n_particles_+1 ; n++ ){
        map_offsets[n] *= (n_measure+1) ;
//        map_offsets_in[n] += n_measurements*n ;
//        DEBUG_VAL(map_offsets[n]) ;
    }
    dev_map_offsets = map_offsets ;

//    DEBUG_MSG("Performing GM reduction") ;
    phdUpdateMergeKernel<<<n_particles_,256>>>
     (raw_pointer_cast(&dev_features_update[0]),
      raw_pointer_cast(&dev_gaussians_merged_tmp[0]),
      raw_pointer_cast(&dev_merged_sizes[0]),
      raw_pointer_cast(&dev_merge_flags[0]),
      raw_pointer_cast(&dev_map_offsets[0]),
            n_particles_, config_.lookup("min_separation")) ;

    // copy out the results of the GM reduction, leaving only valid gaussians
    host_vector<int> merged_sizes = dev_merged_sizes ;
    int n_merged_total = thrust::reduce(merged_sizes.begin(),
                                        merged_sizes.end()) ;
    device_vector<Gaussian6D> dev_gaussians_merged(n_merged_total) ;
    device_vector<Gaussian6D>::iterator it_merged = dev_gaussians_merged.begin() ;
    for ( int n = 0 ; n < merged_sizes.size() ; n++){
        it_merged = thrust::copy_n(&dev_gaussians_merged_tmp[map_offsets[n]],
                        merged_sizes[n],
                        it_merged) ;
    }

    // get the updated feature weights
//    device_vector<float> dev_merged_weights(n_merged_total) ;
    dev_feature_weights_.resize(n_merged_total);
    thrust::transform(dev_gaussians_merged.begin(),
                      dev_gaussians_merged.end(),
                      dev_feature_weights_.begin(),
                      get_weight()) ;

    // ---- Transform features back to Euclidean space ------------------------

    // sample disparity space gaussians
    dev_u_.resize(n_merged_total*particles_per_feature_);
    dev_v_.resize(n_merged_total*particles_per_feature_);
    dev_d_.resize(n_merged_total*particles_per_feature_);
    dev_vu_.resize(n_merged_total*particles_per_feature_);
    dev_vv_.resize(n_merged_total*particles_per_feature_);
    dev_vd_.resize(n_merged_total*particles_per_feature_);
    thrust::for_each(make_zip_iterator(make_tuple(
                                           make_counting_iterator(0),
                                           dev_u_.begin(),
                                           dev_v_.begin(),
                                           dev_d_.begin(),
                                           dev_vu_.begin(),
                                           dev_vv_.begin(),
                                           dev_vd_.begin())),
                     make_zip_iterator(make_tuple(
                                           make_counting_iterator(n_merged_total),
                                           dev_u_.end(),
                                           dev_v_.end(),
                                           dev_d_.end(),
                                           dev_vu_.end(),
                                           dev_vv_.end(),
                                           dev_vd_.end())),
                     sample_disparity_gaussian(
                        raw_pointer_cast(&dev_gaussians_merged[0]),
                        particles_per_feature_
                     )
                ) ;

    int n_feature_particles = particles_per_feature_*n_merged_total ;

    // create particle indices
    dev_particle_indices_.resize(n_feature_particles);
    it_features = dev_particle_indices_.begin() ;
    for ( int n = 0 ; n < n_particles_ ; n++ ){
        it_features = thrust::fill_n(
                    it_features,
                    merged_sizes[n]*particles_per_feature_,
                    n) ;
    }

    // do the transformation
    computeEuclideanParticles() ;

}

void SCPHDCameraCalibration::fitGaussians()
{
    int n_features_total = dev_x_.size()/particles_per_feature_ ;
    int n_blocks = min(cuda_dev_props_.maxGridSize[0],n_features_total) ;

    dev_features_.resize(n_features_total);

    fitGaussiansKernel<<<n_blocks,256>>>
           (raw_pointer_cast(&dev_u_[0]),
            raw_pointer_cast(&dev_v_[0]),
            raw_pointer_cast(&dev_d_[0]),
            raw_pointer_cast(&dev_vu_[0]),
            raw_pointer_cast(&dev_vv_[0]),
            raw_pointer_cast(&dev_vd_[0]),
            raw_pointer_cast(&dev_feature_weights_[0]),
            n_features_total,
            raw_pointer_cast(&dev_features_[0]),
            n_particles_) ;

    // convert particle indices to gaussian indices
    dev_gaussian_indices_.resize(n_features_total) ;
    thrust::gather(make_transform_iterator(make_counting_iterator(0),
                                           times(particles_per_feature_)),
                   make_transform_iterator(make_counting_iterator(n_features_total),
                                           times(particles_per_feature_)),
                   dev_particle_indices_.begin(),
                   dev_gaussian_indices_.begin()) ;
}

void SCPHDCameraCalibration::resample()
{
    host_vector<int> idx_resample(n_particles_) ;
    double interval = 1.0/n_particles_ ;

    thrust::default_random_engine rng ;
    thrust::uniform_real_distribution<float> u01(0,interval) ;
    double r = u01(rng) ;
    double c = exp(particle_weights_[0]) ;
    int i = 0 ;
    for ( int j = 0 ; j < n_particles_ ; j++ )
    {
        r = j*interval + u01(rng)*interval ;
        while( r > c )
        {
            i++ ;
            // sometimes the weights don't exactly add up to 1, so i can run
            // over the indexing bounds. When this happens, find the most highly
            // weighted particle and fill the rest of the new samples with it
            if ( i >= n_particles_ || i < 0 || isnan(i) )
            {
//                DEBUG_VAL(r) ;
//                DEBUG_VAL(c) ;
                double max_weight = -1 ;
                int max_idx = -1 ;
                for ( int k = 0 ; k < n_particles_ ; k++ )
                {
//                    DEBUG_MSG("Warning: particle weights don't add up to 1!s") ;
                    if ( exp(particle_weights_[k]) > max_weight )
                    {
                        max_weight = exp(particle_weights_[k]) ;
                        max_idx = k ;
                    }
                }
                i = max_idx ;
                // set c = 2 so that this while loop is never entered again
                c = 2 ;
                break ;
            }
            c += exp(particle_weights_[i]) ;
        }
        idx_resample[j] = i ;
        r += interval ;
    }

    // resample parent particles
    device_vector<int> dev_idx_resample = idx_resample ;
    device_vector<Extrinsics> dev_new_particles(n_particles_) ;
    thrust::gather(dev_idx_resample.begin(), dev_idx_resample.end(),
                   dev_particle_states_.begin(), dev_new_particles.begin() ) ;

    // resample features
    device_vector<int> dev_map_sizes(n_particles_) ;
    device_vector<int> dev_map_offsets(n_particles_) ;
    host_vector<int> map_sizes(n_particles_) ;
    host_vector<int> map_offsets(n_particles_) ;
    thrust::reduce_by_key(dev_particle_indices_.begin(),
                          dev_particle_indices_.end(),
                          make_constant_iterator(1),
                          make_discard_iterator(),
                          dev_map_sizes.begin()) ;
    thrust::exclusive_scan(dev_map_sizes.begin(), dev_map_sizes.end(),
                           dev_map_offsets.begin()) ;
    map_sizes = dev_map_sizes ;
    map_offsets = dev_map_offsets ;

    int n_feature_particles = dev_x_.size() ;
    device_vector<float> dev_new_x(n_feature_particles) ;
    device_vector<float> dev_new_y(n_feature_particles) ;
    device_vector<float> dev_new_z(n_feature_particles) ;
    device_vector<float> dev_new_vx(n_feature_particles) ;
    device_vector<float> dev_new_vy(n_feature_particles) ;
    device_vector<float> dev_new_vz(n_feature_particles) ;

    device_vector<float>::iterator it_x = dev_new_x.begin() ;
    device_vector<float>::iterator it_y = dev_new_y.begin() ;
    device_vector<float>::iterator it_z = dev_new_z.begin() ;
    device_vector<float>::iterator it_vx = dev_new_vx.begin() ;
    device_vector<float>::iterator it_vy = dev_new_vy.begin() ;
    device_vector<float>::iterator it_vz = dev_new_vz.begin() ;

    for (int i = 0 ; i < n_particles_; i++){
        int idx = idx_resample[i] ;
        int offset = map_offsets[idx] ;
        it_x = thrust::copy_n(dev_x_.begin()+offset,
                              map_sizes[idx],it_x) ;
        it_y = thrust::copy_n(dev_y_.begin()+offset,
                              map_sizes[idx],it_y) ;
        it_z = thrust::copy_n(dev_z_.begin()+offset,
                              map_sizes[idx],it_z) ;
        it_vx = thrust::copy_n(dev_vx_.begin()+offset,
                              map_sizes[idx],it_vx) ;
        it_vy = thrust::copy_n(dev_vy_.begin()+offset,
                              map_sizes[idx],it_vy) ;
        it_vz = thrust::copy_n(dev_vz_.begin()+offset,
                              map_sizes[idx],it_vz) ;
    }

    // save resampled values
    dev_particle_states_ = dev_new_particles ;
    dev_x_ = dev_new_x ;
    dev_y_ = dev_new_y ;
    dev_z_ = dev_new_z ;
    dev_vx_ = dev_new_vx ;
    dev_vy_ = dev_new_vy ;
    dev_vz_ = dev_new_vz ;
}




