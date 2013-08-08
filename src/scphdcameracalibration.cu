#include "scphdcameracalibration.cuh"
#include "kernels.cuh"
#include "device_math.cuh"
//#include "thrust_operators.cu"
#include "types.h"

#include <iostream>
#include <list>
#include <ctime>

#include <matio.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <nvToolsExt.h>
#include <nvToolsExtCuda.h>
#include <nvToolsExtCudaRt.h>

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
#include <thrust/count.h>
#include <thrust/partition.h>
#include <thrust/logical.h>

#ifdef DEBUG
#define DEBUG_MSG(x) cout << "[" << __func__ << "(" << __LINE__ << ")]: " << x << endl
#define DEBUG_VAL(x) cout << "[" << __func__ << "(" << __LINE__ << ")]: " << #x << " = " << x << endl
#else
#define DEBUG_MSG(x)
#define DEBUG_VAL(x)
#endif

using namespace thrust ;
using namespace libconfig ;

//typedef struct{
//    int n ;
//    DisparityPoint mean ;
//    DisparityPoint variance ;
//    double cov_u_v ;
//    double cov_u_d ;
//    double cov_u_vu ;
//    double cov_u_vv ;
//    double cov_u_vd ;
//    double cov_v_d ;
//    double cov_v_vu ;
//    double cov_v_vv ;
//    double cov_v_vd ;
//    double cov_d_vu ;
//    double cov_d_vv ;
//    double cov_d_vd ;
//    double cov_vu_vv ;
//    double cov_vu_vd ;
//    double cov_vv_vd ;

//    void initialize(){
//       DisparityPoint zeros = {0,0,0,0,0,0} ;
//       mean = zeros ;
//       variance = zeros ;
//       n = 0 ;
//       cov_u_v = 0.0 ;
//       cov_u_d = 0.0 ;
//       cov_u_vu = 0.0 ;
//       cov_u_vv = 0.0 ;
//       cov_u_vd = 0.0 ;
//       cov_v_d = 0.0 ;
//       cov_v_vu = 0.0 ;
//       cov_v_vv = 0.0 ;
//       cov_v_vd = 0.0 ;
//       cov_d_vu = 0.0 ;
//       cov_d_vv = 0.0 ;
//       cov_d_vd = 0.0 ;
//       cov_vu_vv = 0.0 ;
//       cov_vu_vd = 0.0 ;
//       cov_vv_vd = 0.0 ;
//    }
//} DisparityStats ;

struct generate_gaussian_noise : public thrust::unary_function<int, double>
{
    thrust::random::normal_distribution<double> dist ;
    double seed_ ;

    generate_gaussian_noise(double seed, double mean = 0.0, double std = 1.0){
        dist = thrust::random::normal_distribution<double>(mean,std) ;
        seed_ = seed ;
    }

    __device__ __host__ double
    operator()(int tid){
        thrust::default_random_engine rng(seed_) ;
        rng.discard(tid);
        return dist(rng) ;
    }
} ;

struct generate_uniform_random : public thrust::unary_function<unsigned int, double>
{
    thrust::random::uniform_real_distribution<double> dist ;
    double seed_ ;

    generate_uniform_random(double seed, double a = 0.0, double b = 1.0){
        dist = thrust::random::uniform_real_distribution<double>(a,b) ;
        seed_ = seed ;
    }

    __device__ __host__ double
    operator()(unsigned int tid){
        thrust::random::default_random_engine rng(seed_) ;
        rng.discard(tid);

        return dist(rng) ;
    }
} ;

struct predict_camera{
    OrientedLinearCVMotionModel3D model_ ;
    double dt ;
    predict_camera(OrientedLinearCVMotionModel3D m, double dt)
        : model_(m), dt(dt) {}

    template <typename T>
    __device__ __host__ void
    operator()(T t){
        Extrinsics state = get<0>(t) ;
        double ax = get<1>(t) ;
        double ay = get<2>(t) ;
        double az = get<3>(t) ;
        double ax_a = get<4>(t) ;
        double ay_a = get<5>(t) ;
        double az_a = get<6>(t) ;


        model_.computeNoisyMotion(state.cartesian,state.angular,
                                  dt,ax,ay,az,ax_a,ay_a,az_a);

        get<0>(t) = state ;
    }
};

struct predict_features{
    const double dt_ ;
    predict_features(double dt) : dt_(dt) {}

    template <typename Tuple>
    __device__ void
    operator()(Tuple t){
        double x = get<0>(t) ;
        double y = get<1>(t) ;
        double z = get<2>(t) ;
        double vx = get<3>(t) ;
        double vy = get<4>(t) ;
        double vz = get<5>(t) ;
        double ax = get<6>(t) ;
        double ay = get<7>(t) ;
        double az = get<8>(t) ;

        get<0>(t) = x + vx*dt_ + 0.5*ax*dt_*dt_ ;
        get<1>(t) = y + vy*dt_ + 0.5*ay*dt_*dt_ ;
        get<2>(t) = z + vz*dt_ + 0.5*az*dt_*dt_ ;
        get<3>(t) = vx + ax*dt_ ;
        get<4>(t) = vy + ay*dt_ ;
        get<5>(t) = vz + az*dt_ ;
    }
};

//struct world_to_disparity_transform{
//    DisparityMeasurementModel model_ ;
//    Extrinsics* extrinsics ;

//    world_to_disparity_transform(DisparityMeasurementModel model,
//                                 Extrinsics* e) :
//        model_(model), extrinsics(e) {}

//    template <typename Tuple>
//    __device__ __host__ tuple<double,double,double,double,double,double,bool>
//    operator()(Tuple t){
//        EuclideanPoint p_world ;
//        p_world.x = get<0>(t) ;
//        p_world.y = get<1>(t) ;
//        p_world.z = get<2>(t) ;
//        p_world.vx = get<3>(t) ;
//        p_world.vy = get<4>(t) ;
//        p_world.vz = get<5>(t) ;
//        int idx = get<6>(t) ;
//        Extrinsics e = extrinsics[idx] ;
//        DisparityPoint p_disparity = model_.computeMeasurement(p_world,e) ;

//        bool in_range = ( p_disparity.u >= 0 ) &&
//                ( p_disparity.u <= model_.img_width() ) &&
//                ( p_disparity.v >= 0 ) &&
//                ( p_disparity.v <= model_.img_height() ) &&
//                ( p_disparity.d >= 0 ) ;
//        return make_tuple(p_disparity.u,
//                        p_disparity.v,
//                        p_disparity.d,
//                        p_disparity.vu,
//                        p_disparity.vv,
//                        p_disparity.vd,
//                          in_range) ;
//    }
//};

//struct disparity_to_world_transform{
//    DisparityMeasurementModel model_ ;
//    Extrinsics* extrinsics ;
//    double max_v ;

//    disparity_to_world_transform(DisparityMeasurementModel model,
//                                 Extrinsics* e, double max_v) :
//        model_(model), extrinsics(e), max_v(max_v) {}

//    template <typename Tuple>
//    __device__ __host__ tuple<double,double,double,double,double,double>
//    operator()(Tuple t){
//        DisparityPoint p_disparity ;
//        p_disparity.u = get<0>(t) ;
//        p_disparity.v = get<1>(t) ;
//        p_disparity.d = get<2>(t) ;
//        p_disparity.vu = get<3>(t) ;
//        p_disparity.vv = get<4>(t) ;
//        p_disparity.vd = get<5>(t) ;
//        Extrinsics e = extrinsics[get<6>(t)] ;

//        EuclideanPoint p_world = model_.invertMeasurement(p_disparity,e) ;

////        if (p_world.vx > max_v)
////            p_world.vx = max_v ;
////        if (p_world.vy > max_v)
////            p_world.vy = max_v ;
////        if (p_world.vz > max_v)
////            p_world.vz = max_v ;

//        return make_tuple(p_world.x,p_world.y, p_world.z,
//                          p_world.vx,p_world.vy,p_world.vz) ;
//    }
//};

//struct pack_disparity_stats :
//    public thrust::unary_function<const DisparityPoint&, DisparityStats>
//{
//    __host__ __device__
//    DisparityStats operator()(const DisparityPoint& p){
//        DisparityStats result ;
//        result.mean = p ;
//        result.n = 1 ;
//        return result ;
//    }
//} ;

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
    double* u_ ;
    double* v_ ;
    double* pd_ ;
    Gaussian6D* features_ ;
    DisparityMeasurementModel model_ ;
    update_components(Gaussian6D* features, double* u, double* v, double* pd,
                           DisparityMeasurementModel model) :
        features_(features), u_(u), v_(v), model_(model), pd_(pd) {}

    template <typename T>
    __device__ void
    operator()(T t){
        // unpack tuple
        int idx_feature = get<0>(t) ;
        int idx_measure = get<1>(t) ;

        Gaussian6D f = features_[idx_feature] ;
        Gaussian6D f_update ;
        double* p = f.cov ;
        double pd = pd_[idx_feature] ;

        double var_u = model_.std_u()*model_.std_u() ;
        double var_v = model_.std_v()*model_.std_v() ;

        // innovation vector
        double innov[2] ;
        innov[0] = u_[idx_measure] - f.mean[0] ;
        innov[1] = v_[idx_measure] - f.mean[1] ;

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
        f_update.weight = safeLog(pd)
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

        get<2>(t) = f_update ;
    }
};

//struct compute_birth{
//    DisparityPoint birth_means ;
//    DisparityPoint birth_vars ;
//    double w0 ;

//    compute_birth(DisparityPoint means0, DisparityPoint vars0, double w0) :
//        birth_means(means0), birth_vars(vars0), w0(w0)
//    {}

//    template<typename T>
//    __device__ void
//    operator()(T t){
//        double u = get<0>(t) ;
//        double v = get<1>(t) ;
//        Gaussian6D feature_birth ;
//        feature_birth.mean[0] = u ;
//        feature_birth.mean[1] = v ;
//        feature_birth.mean[2] = birth_means.d ;
//        feature_birth.mean[3] = birth_means.vu ;
//        feature_birth.mean[4] = birth_means.vv ;
//        feature_birth.mean[5] = birth_means.vd ;

//        for ( int i = 0 ; i < 36 ; i++)
//            feature_birth.cov[i] = 0 ;
//        feature_birth.cov[0] = birth_vars.u ;
//        feature_birth.cov[7] = birth_vars.v ;
//        feature_birth.cov[14] = birth_vars.d ;
//        feature_birth.cov[21] = birth_vars.vu ;
//        feature_birth.cov[28] = birth_vars.vv ;
//        feature_birth.cov[35] = birth_vars.vd ;

//        feature_birth.weight = safeLog(w0) ;

//        get<2>(t) = feature_birth ;
//    }
//};

struct compute_nondetect{

    template <typename T>
    __device__ void
    operator()(T t){
        Gaussian6D feature_predict = get<0>(t) ;
        double pd = get<1>(t) ;

        Gaussian6D feature_nondetect = feature_predict ;
        feature_nondetect.weight = feature_predict.weight*(1-pd) ;
        get<2>(t) = feature_nondetect ;
    }
};


struct log_sum_exp : public thrust::binary_function<const double, const double, double>
{
    __device__ __host__ double
    operator()(const double a, const double b){
        if (a > b){
            return a + safeLog(1 + exp(b-a)) ;
        }
        else{
            return b + safeLog(1 + exp(a-b)) ;
        }
    }
} ;

/// extract the weight
struct get_weight : public thrust::unary_function<const Gaussian6D, double>
{
    __device__ __host__ double
    operator()(const Gaussian6D g){
        return g.weight ;
    }
} ;

/// subtract from a constant
template <typename T>
struct subtract_from : public thrust::unary_function<T,T>
{
    T val_ ;
    __device__ __host__ subtract_from(T val) : val_(val) {}

    __device__ __host__ T
    operator()(T x){ return ( val_ - x) ; }
} ;

/// multiply by 1-pd
struct nondetect_weight : public thrust::binary_function<double,double,double>
{
    __device__ __host__ double
    operator()(double w, double pd){
        return w*(1-pd) ;
    }
} ;


/// multiply by a constant
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

template <typename T>
struct square : public thrust::unary_function<T,T>
{
    __device__ __host__ T
    operator()(T x) { return x*x ; }
} ;

template <typename T>
struct equals : public thrust::unary_function<T,bool>
{
    T val ;
    __device__ __host__ equals(T val) : val(val) {}

    __device__ __host__ bool
    operator()(T x) { return (x == val) ; }
} ;

template <typename T>
struct gt : public thrust::unary_function<T,bool>
{
    T val ;
    __device__ __host__ gt(T val) : val(val) {}

    __device__ __host__ bool
    operator()(T x) { return ( x > val ) ; }
} ;

template <typename T>
struct lt : public thrust::unary_function<T,bool>
{
    T val ;
    __device__ __host__ lt(T val) : val(val) {}

    __device__ __host__ bool
    operator()(T x) { return ( x < val ) ; }
} ;

template <typename T>
struct geq : public thrust::unary_function<T,bool>
{
    T val ;
    __device__ __host__ geq(T val) : val(val) {}

    __device__ __host__ bool
    operator()(T x) { return ( x >= val ) ; }
} ;

template <typename T>
struct leq : public thrust::unary_function<T,bool>
{
    T val ;
    __device__ __host__ leq(T val) : val(val) {}

    __device__ __host__ bool
    operator()(T x) { return ( x <= val ) ; }
} ;

template <typename T>
struct check_nan : public thrust::unary_function<T,bool>
{
    __device__ __host__ bool
    operator()(T x) { return isnan(x) ; }
} ;

struct sample_disparity_gaussian
{
    const int n_samples_ ;
    const Gaussian6D* gaussians_ ;
    double seed ;
    sample_disparity_gaussian(Gaussian6D* g, int n, int s) : n_samples_(n), gaussians_(g), seed(s) {}

    // number of parallel threads is n_features*samples_per_feature
    // tuple argument is (tid, u, v, d, vu, vv, vd) ;
    template<typename T>
    __device__ void
    operator()(T t){
        int tid = get<0>(t) ;

        // generate uncorrelated normally distributed values
        thrust::default_random_engine rng(seed) ;
        rng.discard(6*tid) ;
        thrust::random::normal_distribution<double> randn(0.0,1.0) ;
        double vals[6] ;
        for ( int i = 0 ; i < 6 ; i++)
            vals[i] = randn(rng) ;

//        printf("%d: %f %f %f %f %f %f\n", tid, vals[0],vals[1],vals[2],
//                vals[3],vals[4],vals[5]) ;
        // uncorrelated values are transformed by multiplying with cholesky
        // decomposition of covariance matrix, and adding mean
        int idx = int(tid/n_samples_) ;
        Gaussian6D feature = gaussians_[idx] ;

        double L[36] ;
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

//        if (!isPosDef(feature) && (tid % 99) == 0){
//            printf("non PD matrix\n") ;
//            print_matrix(feature.cov,6) ;
//            print_matrix(L,6);
//        }
    }
};

/// helper function for outputting a Gaussian to std_out
template<class GaussianType>
__host__  void
print_feature(GaussianType f)
{
    int dims = getGaussianDim(f) ;
    cout << "dims: " << dims << endl ;
    cout << "weight: " << f.weight << endl ;
    cout << "mean: " << endl ;
    for ( int i = 0 ; i < dims ; i++ )
        cout << f.mean[i] << " " ;
    cout << endl << "cov: " << endl ;
    for ( int i = 0 ; i < dims ; i++){
        for ( int j = 0 ;j < dims ; j++)
            cout << f.cov[i+j*dims] << " " ;
        cout << endl ;
    }
    cout << endl ;
//#endif
}

template <typename T>
__host__ void print_vector(device_vector<T> vec, int w = 256){
    for ( int i = 0 ; i < vec.size() ; i++){
        T obj = vec[i] ;
        cout << obj << " " ;
        if ( (i % w) == 0  && i > 1 )
            cout << endl ;
    }
    cout << endl ;
}

template <typename T>
__host__ void print_vector(host_vector<T> vec, int w = 256){
    for ( int i = 0 ; i < vec.size() ; i++){
        T obj = vec[i] ;
        cout << obj << " " ;
        if ( (i % w) == 0  && i > 1 )
            cout << endl ;
    }
    cout << endl ;
}

void SCPHDCameraCalibration::transformTest(){
    DEBUG_MSG("Transformation test") ;
    DEBUG_MSG("euclidean particles: ") ;
    cout << "x = " << endl ;
    print_vector(dev_x_) ;
    cout << "y = " << endl ;
    print_vector(dev_y_) ;
    cout << "z = " << endl ;
    print_vector(dev_z_) ;
    cout << "vx = " << endl ;
    print_vector(dev_vx_) ;
    cout << "vy = " << endl ;
    print_vector(dev_vy_) ;
    cout << "vz = " << endl ;
    print_vector(dev_vz_) ;

    DEBUG_MSG("world to disparity (fixed): ") ;
    computeDisparityParticles(true);
    cout << "u = " << endl ;
    print_vector(dev_u_) ;
    cout << "v = " << endl ;
    print_vector(dev_v_) ;
    cout << "d = " << endl ;
    print_vector(dev_d_) ;
    cout << "vu = " << endl ;
    print_vector(dev_vu_) ;
    cout << "vv = " << endl ;
    print_vector(dev_vv_) ;
    cout << "vd = " << endl ;
    print_vector(dev_vd_) ;

    DEBUG_MSG("disparity to world (fixed): ") ;
    computeEuclideanParticles(true);
    cout << "x = " << endl ;
    print_vector(dev_x_) ;
    cout << "y = " << endl ;
    print_vector(dev_y_) ;
    cout << "z = " << endl ;
    print_vector(dev_z_) ;
    cout << "vx = " << endl ;
    print_vector(dev_vx_) ;
    cout << "vy = " << endl ;
    print_vector(dev_vy_) ;
    cout << "vz = " << endl ;
    print_vector(dev_vz_) ;

    DEBUG_MSG("world to disparity (displaced)") ;
    computeDisparityParticles(false);
    cout << "u = " << endl ;
    print_vector(dev_u_) ;
    cout << "v = " << endl ;
    print_vector(dev_v_) ;
    cout << "d = " << endl ;
    print_vector(dev_d_) ;
    cout << "vu = " << endl ;
    print_vector(dev_vu_) ;
    cout << "vv = " << endl ;
    print_vector(dev_vv_) ;
    cout << "vd = " << endl ;
    print_vector(dev_vd_) ;

    DEBUG_MSG("disparity to world (displaced): ") ;
    computeEuclideanParticles(false);
    cout << "x = " << endl ;
    print_vector(dev_x_) ;
    cout << "y = " << endl ;
    print_vector(dev_y_) ;
    cout << "z = " << endl ;
    print_vector(dev_z_) ;
    cout << "vx = " << endl ;
    print_vector(dev_vx_) ;
    cout << "vy = " << endl ;
    print_vector(dev_vy_) ;
    cout << "vz = " << endl ;
    print_vector(dev_vz_) ;
}

void SCPHDCameraCalibration::checkStuff()
{
    // check for nan weights
    for ( int i = 0 ; i < particle_weights_.size() ; i++ ){
        if (particle_weights_[i] != particle_weights_[i]){
            cout << "nan weight detected" << endl ;
            exit(2);
        }
    }
}

SCPHDCameraCalibration::SCPHDCameraCalibration(const char* config_file)
{
    initializeCuda() ;

    config_.readFile(config_file);

    verbosity_ = config_.lookup("verbosity") ;

    cout << "configuring particle counts" << endl ;
    n_particles_ = config_.lookup("n_particles") ;
    particles_per_feature_ = config_.lookup("particles_per_feature") ;

    // initialize particle weights
    particle_weights_.resize(n_particles_,1.0/n_particles_);

    DEBUG_MSG("initial particle weights") ;
    print_vector(particle_weights_) ;

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
    thrust::random::normal_distribution<double> randn(0.0,1.0) ;
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

    const Setting& camera_motion_config = config_.lookup("camera_motion_model") ;
    camera_motion_model_ = OrientedLinearCVMotionModel3D(
            camera_motion_config["cartesian"]["ax"],
            camera_motion_config["cartesian"]["ay"],
            camera_motion_config["cartesian"]["az"],
            camera_motion_config["angular"]["ax"],
            camera_motion_config["angular"]["ay"],
            camera_motion_config["angular"]["az"]
            ) ;

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
}

void SCPHDCameraCalibration::initializeCuda(){
    int n_cuda_devices = 0 ;
    checkCudaErrors( cudaGetDeviceCount(&n_cuda_devices) ) ;
    std::cout << "Found " << n_cuda_devices << " CUDA devices" << std::endl ;

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
    cout << "Selected " << cuda_dev_props_.name << endl ;

    checkCudaErrors( cudaDeviceReset() ) ;

    // trick to establish context
    cudaFree(0) ;
}

void SCPHDCameraCalibration::predict(double dt)
{
    //---------------------------- predict calibration -----------------------

    int n_samples = n_particles_ ;
    device_vector<double> ax(n_samples) ;
    device_vector<double> ay(n_samples) ;
    device_vector<double> az(n_samples) ;
    device_vector<double> ax_a(n_samples) ;
    device_vector<double> ay_a(n_samples) ;
    device_vector<double> az_a(n_samples) ;

    DEBUG_MSG("generate gaussian noise") ;
    time_t time_val ;
    time(&time_val) ;
    DEBUG_VAL(time_val) ;

//    thrust::random::default_random_engine rng(time_val) ;
//    thrust::random::normal_distribution<double> randn(0.0,1.0) ;
//    for ( int i = 0 ; i < n_samples ; i++ ){
//        ax[i] = randn(rng)*camera_motion_model_.std_ax_cartesian() ;
//        ay[i] = randn(rng)*camera_motion_model_.std_ay_cartesian() ;
//        az[i] = randn(rng)*camera_motion_model_.std_az_cartesian() ;
//        ax_a[i] = randn(rng)*camera_motion_model_.std_ax_angular() ;
//        ay_a[i] = randn(rng)*camera_motion_model_.std_ay_angular() ;
//        az_a[i] = randn(rng)*camera_motion_model_.std_az_angular() ;
//    }

    nvtxRangeId_t predict_id = nvtxRangeStartA("generate noise camera") ;
    DEBUG_MSG("ax") ;
    thrust::transform(counting_iterator<int>(0),counting_iterator<int>(n_samples),
                      ax.begin(),generate_uniform_random(time_val,0,camera_motion_model_.std_ax_cartesian())) ;

    thrust::transform(counting_iterator<int>(0),counting_iterator<int>(n_samples),
                      ax.begin(),generate_gaussian_noise(time_val,0,camera_motion_model_.std_ax_cartesian())) ;
    time(&time_val) ;
    DEBUG_MSG("ay") ;
    thrust::transform(counting_iterator<int>(0),counting_iterator<int>(n_samples),
                      ay.begin(),generate_gaussian_noise(time_val,0,camera_motion_model_.std_ay_cartesian())) ;
    time(&time_val) ;
    DEBUG_MSG("az") ;
    thrust::transform(counting_iterator<int>(0),counting_iterator<int>(n_samples),
                      az.begin(),generate_gaussian_noise(time_val,0,camera_motion_model_.std_az_cartesian())) ;
    time(&time_val) ;
    DEBUG_MSG("ax_a") ;
    thrust::transform(counting_iterator<int>(0),counting_iterator<int>(n_samples),
                      ax_a.begin(),generate_gaussian_noise(time_val,0,camera_motion_model_.std_ax_angular())) ;
    time(&time_val) ;
    DEBUG_MSG("ay_a") ;
    thrust::transform(counting_iterator<int>(0),counting_iterator<int>(n_samples),
                      ay_a.begin(),generate_gaussian_noise(time_val,0,camera_motion_model_.std_ay_angular())) ;
    time(&time_val) ;
    DEBUG_MSG("az_a") ;
    thrust::transform(counting_iterator<int>(0),counting_iterator<int>(n_samples),
                      az_a.begin(),generate_gaussian_noise(time_val,0,camera_motion_model_.std_az_angular())) ;
    nvtxRangeEnd(predict_id); ;

    predict_id = nvtxRangeStartA("predict camera") ;
    DEBUG_MSG("predict camera") ;
    thrust::for_each(make_zip_iterator(make_tuple(dev_particle_states_.begin(),
                                                  ax.begin(),ay.begin(),az.begin(),
                                                  ax_a.begin(),ay_a.begin(),az_a.begin())),
                     make_zip_iterator(make_tuple(dev_particle_states_.end(),
                                                ax.end(),ay.end(),az.end(),
                                                ax_a.end(),ay_a.end(),az_a.end())),
                     predict_camera(camera_motion_model_,dt)) ;
    nvtxRangeEnd(predict_id);

    //---------------------------- predict features --------------------------

    n_samples = dev_x_.size() ;
    ax.resize(n_samples);
    ay.resize(n_samples);
    az.resize(n_samples);

    predict_id = nvtxRangeStartA("generate noise features") ;
    time(&time_val) ;
    thrust::transform(counting_iterator<int>(0), counting_iterator<int>(n_samples),
                      ax.begin(),generate_gaussian_noise(int(time_val),0,feature_motion_model_.std_ax())) ;
    time(&time_val) ;
    thrust::transform(counting_iterator<int>(0), counting_iterator<int>(n_samples),
                      ay.begin(),generate_gaussian_noise(int(time_val),0,feature_motion_model_.std_ay())) ;
    time(&time_val) ;
    thrust::transform(counting_iterator<int>(0), counting_iterator<int>(n_samples),
                      az.begin(),generate_gaussian_noise(int(time_val),0,feature_motion_model_.std_az())) ;
    nvtxRangeEnd(predict_id);

    predict_id = nvtxRangeStartA("predict features") ;
    DEBUG_MSG("predict features") ;
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
                     predict_features(dt)) ;
    nvtxRangeEnd(predict_id);

    predict_id = nvtxRangeStartA("reweight with ps") ;
    double ps = config_.lookup("ps") ;
    thrust::transform(dev_feature_weights_.begin(),dev_feature_weights_.end(),
                      make_constant_iterator<double>(ps),
                      dev_feature_weights_.begin(),
                      thrust::multiplies<double>()) ;
    nvtxRangeEnd(predict_id);
}



void SCPHDCameraCalibration::computeDisparityParticles(bool fixed_camera = false){
    // make sure there is enough space to store disparity particles
    int n_particles = dev_x_.size() ;
    dev_u_.resize(n_particles);
    dev_v_.resize(n_particles);
    dev_d_.resize(n_particles);
    dev_vu_.resize(n_particles);
    dev_vv_.resize(n_particles);
    dev_vd_.resize(n_particles);

    DEBUG_VAL(n_particles) ;

    device_vector<Extrinsics> e ;
    if (!fixed_camera){
        cout << "transforming with translated camera" << endl ;
        e = dev_particle_states_ ;
    }
    else{
        cout << "transforming with fixed camera" << endl ;
        host_vector<Extrinsics> h_e(n_particles_) ;
        e = h_e ;
    }

    dev_particle_pd_.resize(n_particles) ;

    int n_threads = 128 ;
    int n_blocks = ceil(double(n_particles)/n_threads) ;
    DEBUG_MSG("worldToDisparityKernel") ;
    worldToDisparityKernel<<<n_blocks,n_threads>>>(
            raw_pointer_cast(&dev_x_[0]),raw_pointer_cast(&dev_y_[0]),
            raw_pointer_cast(&dev_z_[0]),raw_pointer_cast(&dev_vx_[0]),
            raw_pointer_cast(&dev_vy_[0]),raw_pointer_cast(&dev_vz_[0]),
            measurement_model_, raw_pointer_cast(&e[0]),
            raw_pointer_cast(&dev_particle_indices_[0]), n_particles,
            raw_pointer_cast(&dev_u_[0]),raw_pointer_cast(&dev_v_[0]),
            raw_pointer_cast(&dev_d_[0]),raw_pointer_cast(&dev_vu_[0]),
            raw_pointer_cast(&dev_vv_[0]),raw_pointer_cast(&dev_vd_[0]),
            raw_pointer_cast(&dev_particle_pd_[0])) ;
    cudaDeviceSynchronize() ;

    // compute per-feature pd by taking arithmetic mean of particle pds
    nvtxRangeId_t id = nvtxRangeStartA("compute per-feature pd") ;
    int n_features = n_particles/particles_per_feature_ ;
    dev_pd_.resize(n_features);
    n_threads = 256 ;
    n_blocks = min(n_features,cuda_dev_props_.maxGridSize[0]) ;
    computePdKernel<<<n_blocks,n_threads>>>(
            raw_pointer_cast(&dev_particle_pd_[0]),
            particles_per_feature_,n_features,
            raw_pointer_cast(&dev_pd_[0]));

    cudaDeviceSynchronize() ;
//    device_vector<int> dev_keys(n_particles) ;
//    device_vector<int>::iterator keys_it = dev_keys.begin() ;
//    for ( int n = 0 ; n < n_features ; n++ ){
//        keys_it = thrust::fill_n(keys_it,particles_per_feature_,n) ;
//    }

//    thrust::reduce_by_key(dev_keys.begin(),dev_keys.end(),dev_particle_pd_.begin(),
//                          make_discard_iterator(),dev_pd_.begin()) ;
//    thrust::transform(dev_pd_.begin(),dev_pd_.end(),dev_pd_.begin(),divide_by<double,double>(double(particles_per_feature_))) ;

    nvtxRangeEnd(id);
    //TODO: debug only
//    print_vector(dev_pd_) ;
}

void SCPHDCameraCalibration::computeEuclideanParticles(bool fixed_camera = false){
    // make sure there is enough space to store the particles
    int n_particles = dev_u_.size() ;

    nvtxRangeId_t resize_id = nvtxRangeStartA("computeEuclidean resize") ;
    dev_x_.resize(n_particles);
    dev_y_.resize(n_particles);
    dev_z_.resize(n_particles);
    dev_vx_.resize(n_particles);
    dev_vy_.resize(n_particles);
    dev_vz_.resize(n_particles);
    nvtxRangeEnd(resize_id);

    device_vector<Extrinsics> e ;
    if (!fixed_camera){
        cout << "transforming with translated camera" << endl ;
        e = dev_particle_states_ ;
    }
    else{
        cout << "transforming with fixed camera" << endl ;
        host_vector<Extrinsics> h_e(n_particles_) ;
        e = h_e ;
    }

    double max_v = config_.lookup("max_velocity") ;
    double min_z = config_.lookup("min_range") ;

    int n_threads = 128 ;
    int n_blocks = ceil(double(n_particles)/n_threads) ;
    disparityToWorldKernel<<<n_blocks,n_threads>>>(
            raw_pointer_cast(&dev_u_[0]),raw_pointer_cast(&dev_v_[0]),
            raw_pointer_cast(&dev_d_[0]),raw_pointer_cast(&dev_vu_[0]),
            raw_pointer_cast(&dev_vv_[0]),raw_pointer_cast(&dev_vd_[0]),
            measurement_model_, raw_pointer_cast(&e[0]),
            raw_pointer_cast(&dev_particle_indices_[0]), n_particles,
            raw_pointer_cast(&dev_x_[0]),raw_pointer_cast(&dev_y_[0]),
            raw_pointer_cast(&dev_z_[0]),raw_pointer_cast(&dev_vx_[0]),
            raw_pointer_cast(&dev_vy_[0]),raw_pointer_cast(&dev_vz_[0])) ;
    cudaDeviceSynchronize() ;

//    thrust::transform(make_zip_iterator(make_tuple(dev_u_.begin(),
//                                                   dev_v_.begin(),
//                                                   dev_d_.begin(),
//                                                   dev_vu_.begin(),
//                                                   dev_vv_.begin(),
//                                                   dev_vd_.begin(),
//                                                   dev_particle_indices_.begin())),
//                      make_zip_iterator(make_tuple(dev_u_.end(),
//                                                   dev_v_.end(),
//                                                   dev_d_.end(),
//                                                   dev_vu_.end(),
//                                                   dev_vv_.end(),
//                                                   dev_vd_.end(),
//                                                   dev_particle_indices_.end())),
//                      make_zip_iterator(make_tuple(dev_x_.begin(),
//                                                   dev_y_.begin(),
//                                                   dev_z_.begin(),
//                                                   dev_vx_.begin(),
//                                                   dev_vy_.begin(),
//                                                   dev_vz_.begin())),
//                      disparity_to_world_transform(measurement_model_,
//                                                   raw_pointer_cast(&e[0]),
//                                                    max_v)
    //            );
}


/// reweight disparity particles by pd and resample
void SCPHDCameraCalibration::computeNonDetections()
{
    int n_particles = dev_u_.size() ;
    int n_features = n_particles/particles_per_feature_ ;

    // allocate memory for resampled features
    nvtxRangeId_t nondetect_id = nvtxRangeStartA("nondetect resize") ;
    dev_u_nondetect_.resize(n_particles) ;
    dev_v_nondetect_.resize(n_particles) ;
    dev_d_nondetect_.resize(n_particles) ;
    dev_vu_nondetect_.resize(n_particles) ;
    dev_vv_nondetect_.resize(n_particles) ;
    dev_vd_nondetect_.resize(n_particles) ;
    dev_particle_weights_nondetect_.resize(n_particles);
    dev_indices_nondetect_.resize(n_particles);

    dev_feature_weights_nondetect_.resize(n_features);
    nvtxRangeEnd(nondetect_id);

    nondetect_id = nvtxRangeStartA("nondetect weights") ;
    DEBUG_MSG("prune and copy non-detection terms") ;
//    print_vector(dev_feature_weights_) ;
//    print_vector(dev_pd_) ;

    thrust::transform(dev_feature_weights_.begin(),
                      dev_feature_weights_.end(),
                      dev_pd_.begin(),
                      dev_feature_weights_nondetect_.begin(),
                      nondetect_weight()) ;

    DEBUG_MSG("expand feature weight by number of particles") ;

    host_vector<double> h_weights = dev_feature_weights_nondetect_ ;
//    print_vector(h_weights) ;
    host_vector<double> h_particle_weights(n_particles) ;
    host_vector<double>::iterator it = h_particle_weights.begin() ;
    for ( int n = 0; n < n_features ; n++ ){
        it = thrust::fill_n(it,particles_per_feature_,h_weights[n]) ;
    }

    device_vector<double> d_particle_weights = h_particle_weights ;

    nvtxRangeEnd(nondetect_id);
    // keep if w*(1-pd) >= x ==> pd < (1-x)
    double min_weight = config_.lookup("min_weight") ;


    nondetect_id = nvtxRangeStartA("nondetect prune") ;
    DEBUG_MSG("thrust::copy_if") ;
    device_vector<double>::iterator it_d ;

    // particle states
    it_d = thrust::copy_if(dev_u_.begin(),dev_u_.end(),d_particle_weights.begin(),
                    dev_u_nondetect_.begin(), geq<double>(min_weight)) ;
    thrust::copy_if(dev_v_.begin(),dev_v_.end(),d_particle_weights.begin(),
                    dev_v_nondetect_.begin(), geq<double>(min_weight)) ;
    thrust::copy_if(dev_d_.begin(),dev_d_.end(),d_particle_weights.begin(),
                    dev_d_nondetect_.begin(), geq<double>(min_weight)) ;
    thrust::copy_if(dev_vu_.begin(),dev_vu_.end(),d_particle_weights.begin(),
                    dev_vu_nondetect_.begin(), geq<double>(min_weight)) ;
    thrust::copy_if(dev_vv_.begin(),dev_vv_.end(),d_particle_weights.begin(),
                    dev_vv_nondetect_.begin(), geq<double>(min_weight)) ;
    thrust::copy_if(dev_vd_.begin(),dev_vd_.end(),d_particle_weights.begin(),
                    dev_vd_nondetect_.begin(), geq<double>(min_weight)) ;

    // particle weights
    thrust::copy_if(dev_particle_pd_.begin(), dev_particle_pd_.end(),
                    d_particle_weights.begin(),
                    dev_particle_weights_nondetect_.begin(),
                    geq<double>(min_weight)) ;

    // particle indices
    thrust::copy_if(dev_particle_indices_.begin(), dev_particle_indices_.end(),
                    d_particle_weights.begin(),
                    dev_indices_nondetect_.begin(),
                    geq<double>(min_weight)) ;

    // macro feature weights
    thrust::remove_if(dev_feature_weights_nondetect_.begin(),
                      dev_feature_weights_nondetect_.end(),
                      lt<double>(min_weight)) ;

    int n_pruned = it_d - dev_u_nondetect_.begin() ;
    nvtxRangeEnd(nondetect_id);

    DEBUG_MSG("shrink vectors to pruned size") ;
    dev_u_nondetect_.resize(n_pruned);
    dev_u_nondetect_.shrink_to_fit();

    dev_v_nondetect_.resize(n_pruned);
    dev_v_nondetect_.shrink_to_fit();

    dev_d_nondetect_.resize(n_pruned);
    dev_d_nondetect_.shrink_to_fit();

    dev_vu_nondetect_.resize(n_pruned);
    dev_vu_nondetect_.shrink_to_fit();

    dev_vv_nondetect_.resize(n_pruned);
    dev_vv_nondetect_.shrink_to_fit();

    dev_vd_nondetect_.resize(n_pruned);
    dev_vd_nondetect_.shrink_to_fit();

    dev_particle_weights_nondetect_.resize(n_pruned);
    dev_particle_weights_nondetect_.shrink_to_fit();

    dev_indices_nondetect_.resize(n_pruned);
    dev_indices_nondetect_.shrink_to_fit();

    dev_feature_weights_nondetect_.resize(n_pruned/particles_per_feature_);
    dev_feature_weights_nondetect_.shrink_to_fit();

    DEBUG_VAL(n_pruned/particles_per_feature_) ;

    // nondetect particle weights should be (1-pd)
    thrust::transform(dev_particle_weights_nondetect_.begin(),
                      dev_particle_weights_nondetect_.end(),
                      dev_particle_weights_nondetect_.begin(),
                      subtract_from<double>(1.0)) ;

}

void SCPHDCameraCalibration::recombineNonDetections()
{
    if (dev_u_nondetect_.size() > 0){
        int n_combined = dev_u_.size() + dev_u_nondetect_.size() ;
        device_vector<double> d_u_combined(n_combined) ;
        device_vector<double> d_v_combined(n_combined) ;
        device_vector<double> d_d_combined(n_combined) ;
        device_vector<double> d_vu_combined(n_combined) ;
        device_vector<double> d_vv_combined(n_combined) ;
        device_vector<double> d_vd_combined(n_combined) ;
        device_vector<double> d_particle_weights_combined(n_combined) ;
        device_vector<double> d_feature_weights_combined(n_combined/particles_per_feature_ ) ;


        device_vector<int> d_offsets_nondetect = computeMapOffsets(dev_indices_nondetect_) ;
        device_vector<int> d_offsets = computeMapOffsets(dev_particle_indices_) ;

        if ( verbosity_ >= 3){
            DEBUG_MSG("offsets ") ;
            print_vector(d_offsets) ;

            DEBUG_MSG("offsets nondetect") ;
            print_vector(d_offsets_nondetect) ;

            DEBUG_MSG("particle weights: ") ;
            print_vector(dev_particle_weights_) ;

            DEBUG_MSG("particle weights nondetect: ") ;
            print_vector(dev_particle_weights_nondetect_) ;
        }

        // resize index vector
        dev_particle_indices_.resize(n_combined);
        dev_gaussian_indices_.resize(n_combined/particles_per_feature_);

        // launch the kernel
        int n_threads = 128 ;
        int n_blocks = ceil(double(n_combined)/n_threads) ;
        DEBUG_VAL(n_blocks) ;

        DEBUG_MSG("interleaveKernel") ;
        nvtxRangeId_t interleave_id = nvtxRangeStartA("Interleave Kernel") ;

        // particle weights
        interleaveKernel<<<n_blocks,n_threads>>>(
                raw_pointer_cast(&dev_particle_weights_[0]),
                raw_pointer_cast(&dev_particle_weights_nondetect_[0]),
                raw_pointer_cast(&d_offsets[0]),
                raw_pointer_cast(&d_offsets_nondetect[0]),
                n_particles_, n_combined,
                raw_pointer_cast(&d_particle_weights_combined[0]),
                raw_pointer_cast(&dev_particle_indices_[0])) ;
        cudaDeviceSynchronize() ;

        // particle states
        interleaveKernel<<<n_blocks,n_threads>>>(
                raw_pointer_cast(&dev_u_[0]),
                raw_pointer_cast(&dev_u_nondetect_[0]),
                raw_pointer_cast(&d_offsets[0]),
                raw_pointer_cast(&d_offsets_nondetect[0]),
                n_particles_,n_combined,
                raw_pointer_cast(&d_u_combined[0]),
                raw_pointer_cast(&dev_particle_indices_[0])) ;
        cudaDeviceSynchronize() ;

        interleaveKernel<<<n_blocks,n_threads>>>(
                raw_pointer_cast(&dev_v_[0]),
                raw_pointer_cast(&dev_v_nondetect_[0]),
                raw_pointer_cast(&d_offsets[0]),
                raw_pointer_cast(&d_offsets_nondetect[0]),
                n_particles_,n_combined,
                raw_pointer_cast(&d_v_combined[0]),
                raw_pointer_cast(&dev_particle_indices_[0])) ;
        cudaDeviceSynchronize() ;

        interleaveKernel<<<n_blocks,n_threads>>>(
                raw_pointer_cast(&dev_d_[0]),
                raw_pointer_cast(&dev_d_nondetect_[0]),
                raw_pointer_cast(&d_offsets[0]),
                raw_pointer_cast(&d_offsets_nondetect[0]),
                n_particles_,n_combined,
                raw_pointer_cast(&d_d_combined[0]),
                raw_pointer_cast(&dev_particle_indices_[0])) ;
        cudaDeviceSynchronize() ;

        interleaveKernel<<<n_blocks,n_threads>>>(
                raw_pointer_cast(&dev_vu_[0]),
                raw_pointer_cast(&dev_vu_nondetect_[0]),
                raw_pointer_cast(&d_offsets[0]),
                raw_pointer_cast(&d_offsets_nondetect[0]),
                n_particles_,n_combined,
                raw_pointer_cast(&d_vu_combined[0]),
                raw_pointer_cast(&dev_particle_indices_[0])) ;
        cudaDeviceSynchronize() ;

        interleaveKernel<<<n_blocks,n_threads>>>(
                raw_pointer_cast(&dev_vv_[0]),
                raw_pointer_cast(&dev_vv_nondetect_[0]),
                raw_pointer_cast(&d_offsets[0]),
                raw_pointer_cast(&d_offsets_nondetect[0]),
                n_particles_,n_combined,
                raw_pointer_cast(&d_vv_combined[0]),
                raw_pointer_cast(&dev_particle_indices_[0])) ;
        cudaDeviceSynchronize() ;

        interleaveKernel<<<n_blocks,n_threads>>>(
                raw_pointer_cast(&dev_vd_[0]),
                raw_pointer_cast(&dev_vd_nondetect_[0]),
                raw_pointer_cast(&d_offsets[0]),
                raw_pointer_cast(&d_offsets_nondetect[0]),
                n_particles_,n_combined,
                raw_pointer_cast(&d_vd_combined[0]),
                raw_pointer_cast(&dev_particle_indices_[0])) ;
        cudaDeviceSynchronize() ;

        nvtxRangeEnd(interleave_id);


        // combine macro feature weights
        n_blocks = ceil(double(n_blocks)/particles_per_feature_) ;

        thrust::transform(d_offsets.begin(),
                          d_offsets.end(),
                          d_offsets.begin(),
                          divide_by<int,int>(particles_per_feature_)) ;

        thrust::transform(d_offsets_nondetect.begin(),
                          d_offsets_nondetect.end(),
                          d_offsets_nondetect.begin(),
                          divide_by<int,int>(particles_per_feature_)) ;

        interleaveKernel<<<n_blocks,n_threads>>>(
                raw_pointer_cast(&dev_feature_weights_[0]),
                raw_pointer_cast(&dev_feature_weights_nondetect_[0]),
                raw_pointer_cast(&d_offsets[0]),
                raw_pointer_cast(&d_offsets_nondetect[0]),
                n_particles_, n_combined/particles_per_feature_,
                raw_pointer_cast(&d_feature_weights_combined[0]),
                raw_pointer_cast(&dev_gaussian_indices_[0])
                ) ;
        cudaDeviceSynchronize() ;

        dev_u_ = d_u_combined ;
        dev_v_ = d_v_combined ;
        dev_d_ = d_d_combined ;
        dev_vu_ = d_vu_combined ;
        dev_vv_ = d_vv_combined ;
        dev_vd_ = d_vd_combined ;
        dev_particle_weights_ = d_particle_weights_combined ;
        dev_feature_weights_ = d_feature_weights_combined ;
    }
}

void SCPHDCameraCalibration::update(vector<double> u, vector<double> v,
                                    bool fixed_camera = false){
    DEBUG_MSG("transform euclidean particles to disparity space") ;

//    cout << "euclidean particles: " << endl ;
//    for ( int i = 0 ; i < dev_x_.size() ; i++){
//        cout << dev_x_[i] << " " << dev_y_[i] << " " << dev_z_[i] << " "
//             << dev_vx_[i] << " " << dev_vy_[i] << " " << dev_vz_[i] << endl ;
//    }

    computeDisparityParticles(fixed_camera) ;


//    cout << "disparity particles: " << endl ;
//    for ( int i = 0 ; i < dev_u_.size() ; i++){
//        cout << dev_u_[i] << " " << dev_v_[i] << " " << dev_d_[i] << " "
//             << dev_vu_[i] << " " << dev_vv_[i] << " " << dev_vd_[i] << endl ;
//    }

//    DEBUG_MSG("compute feature weights = old_weight*feature_pd") ;
//    thrust::transform(dev_feature_weights_.begin(),
//                      dev_feature_weights_.end(),
//                      dev_pd_.begin(),
//                      dev_feature_weights_.begin(),
//                      thrust::multiplies<double>()) ;

    nvtxRangeId_t update_id = nvtxRangeStartA("weight particles by pd") ;
    DEBUG_MSG("multiply particle weights by pd") ;
    thrust::transform(dev_particle_weights_.begin(),
                      dev_particle_weights_.end(),
                      dev_particle_pd_.begin(),
                      dev_particle_weights_.begin(),
                      thrust::multiplies<double>()) ;
    nvtxRangeEnd(update_id);

    DEBUG_MSG("fit gaussians") ;
    fitGaussians() ;

//    DEBUG_MSG("remove out of range features") ;
//    separateOutOfRange();

    int n_measure = u.size() ;
    int n_features = dev_features_.size() ;
    int n_detect = n_measure*n_features ;

    DEBUG_VAL(n_measure) ;
    DEBUG_VAL(n_features) ;
    DEBUG_VAL(n_detect) ;

    if (config_.lookup("save_all_maps")){
        dev_features_predicted_.resize(n_features);
        thrust::copy(dev_features_.begin(),dev_features_.end(),
                     dev_features_predicted_.begin()) ;
        gaussian_indices_predicted_ = dev_gaussian_indices_ ;
    }

    DEBUG_MSG("compute map offsets") ;
    device_vector<int> dev_map_offsets =
            computeMapOffsets(dev_gaussian_indices_) ;
//    host_vector<int> map_sizes ;
//    if (dev_gaussian_indices_.size() > 0){
//        device_vector<int> dev_map_sizes(n_particles_) ;
//        thrust::reduce_by_key(dev_gaussian_indices_.begin(),
//                              dev_gaussian_indices_.end(),
//                              make_constant_iterator(1),
//                              make_discard_iterator(),
//                              dev_map_sizes.begin()) ;
//        map_sizes = dev_map_sizes ;
//    }
//    else{
//        map_sizes.resize(n_particles_,0);
//    }

//    // compute the indexing offsets with an inclusive scan
//    host_vector<int> map_offsets(n_particles_+ 1) ;
//    map_offsets[0] = 0 ;
//    thrust::inclusive_scan(map_sizes.begin(),map_sizes.end(),
//                           map_offsets.begin()+1) ;
//    device_vector<int> dev_map_offsets = map_offsets ;



    DEBUG_MSG("copy measurements to device") ;
    device_vector<double> dev_measure_u = u ;
    device_vector<double> dev_measure_v = v ;

    print_vector(dev_measure_u) ;
    print_vector(dev_measure_v) ;

    DEBUG_MSG("compute birth terms") ;


    // there will probably never be more than a few hundred measurements per scan,
    // so just compute the birth features on the host.
    device_vector<Gaussian6D> dev_features_birth = computeBirths(u,v);


//    thrust::for_each(make_zip_iterator(make_tuple(
//                        dev_measure_u.begin(),
//                        dev_measure_v.begin(),
//                        dev_features_birth.begin())),
//                     make_zip_iterator(make_tuple(
//                       dev_measure_u.end(),
//                       dev_measure_v.end(),
//                       dev_features_birth.end())),
//                     compute_birth(birth_mean,birth_vars,birth_weight)) ;

//    cout << "birth features" << endl ;
//    for ( int i = 0 ; i < dev_features_birth.size() ; i++){
//        Gaussian6D f = dev_features_birth[i] ;
//        print_feature(f) ;
//    }

    DEBUG_MSG("compute non-detection terms") ;
    computeNonDetections();

    DEBUG_MSG("compute detection terms") ;

    int n_threads = 128 ;

    dim3 nb ;
    nb.x = ceil((double)n_features/n_threads) ;
    nb.y = n_measure ;
    nb.z = 1 ;

    DEBUG_VAL(nb.x) ;
    DEBUG_VAL(nb.y) ;

    // vector for storing measurement index associated with each detection term,
    // to be used later when computing normalizers via thrust::reduce_by_key
    device_vector<int> dev_idx_measure(n_detect) ;
    device_vector<Gaussian6D> dev_features_detect(n_detect) ;

    computeDetectionsKernel<<<nb, n_threads>>>(
            raw_pointer_cast(&dev_features_[0]),
            raw_pointer_cast(&dev_measure_u[0]),
            raw_pointer_cast(&dev_measure_v[0]),
            raw_pointer_cast(&dev_pd_[0]),
            measurement_model_,
            raw_pointer_cast(&dev_map_offsets[0]),
            n_features, n_measure,
            raw_pointer_cast(&dev_features_detect[0]),
            raw_pointer_cast(&dev_idx_measure[0]));
    cudaDeviceSynchronize() ;

//    cout << "detection features" << endl ;
//    for ( int i = 0 ; i < dev_features_detect.size() ; i++){
//        Gaussian6D f = dev_features_detect[i] ;
//        print_feature(f) ;
//    }


    DEBUG_MSG("compute normalization terms") ;
    nvtxRangeId_t normalizer_range_id = nvtxRangeStartA("compute normalizers") ;
    // use thrust::reduce_by_key to sum only over terms from the same map,
    // and the same measurement

    // extract feature weights
    device_vector<double> dev_weights_detect(n_detect) ;
    thrust::transform(dev_features_detect.begin(),dev_features_detect.end(),
                      dev_weights_detect.begin(),get_weight()) ;
//    print_vector(dev_weights_detect) ;


    // consecutive keys are features from the same measurement
    int n_normalizers = n_particles_*n_measure ;
    device_vector<double> dev_normalizers(n_normalizers) ;
    double birth_plus_clutter = safeLog(n_measure*double(config_.lookup("birth.w0")) +
                                       measurement_model_.kappa()) ;
//    print_vector(dev_idx_measure) ;
    if ( n_features > 0){
        DEBUG_VAL(birth_plus_clutter) ;
        thrust::equal_to<double> binary_pred ;
        thrust::reduce_by_key(dev_idx_measure.begin(),
                              dev_idx_measure.end(),
                              dev_weights_detect.begin(),
                              make_discard_iterator(),
                              dev_normalizers.begin(),
                              binary_pred,
                              log_sum_exp()) ;
//        print_vector(dev_normalizers) ;
        thrust::transform(dev_normalizers.begin(),
                          dev_normalizers.end(),
                          thrust::constant_iterator<double>(birth_plus_clutter),
                          dev_normalizers.begin(),
                          log_sum_exp()) ;
    }
    else{
        thrust::fill(dev_normalizers.begin(),dev_normalizers.end(),
                     birth_plus_clutter) ;
    }
    nvtxRangeEnd(normalizer_range_id) ;

    if (verbosity_ >= 3){
        DEBUG_MSG("normalizer values") ;
        print_vector(dev_normalizers) ;
    }

    // ------- SCPHD update ---------------------------------------------- //

    // allocate space for updated features
    int n_update = n_features*n_measure + n_measure*n_particles_ ;
    DEBUG_VAL(n_update) ;
    device_vector<Gaussian6D> dev_features_update(n_update);


    // create vector of flags to control GM merging
    device_vector<bool> dev_merge_flags(n_update) ;

    DEBUG_MSG("launch scphd update kernel") ;
    updateKernel<<<n_particles_,256>>>(
            raw_pointer_cast(&dev_features_detect[0]),
            raw_pointer_cast(&dev_features_birth[0]),
            raw_pointer_cast(&dev_normalizers[0]),
            raw_pointer_cast(&dev_map_offsets[0]),
            n_particles_, n_measure,
            raw_pointer_cast(&dev_features_update[0]),
            raw_pointer_cast(&dev_merge_flags[0]),
            config_.lookup("min_weight"));
    cudaDeviceSynchronize() ;

    // free memory
    dev_features_detect.resize(0);
    dev_features_detect.shrink_to_fit();

    dev_features_birth.resize(0);
    dev_features_birth.shrink_to_fit();

//    DEBUG_MSG("Updated gaussians and merge flags: ") ;
//    for (int n = 0 ; n < n_update ; n++){
//        bool flag = dev_merge_flags[n] ;
//        cout << flag << " " ;
//        Gaussian6D g = dev_features_update[n] ;
//        print_feature(g) ;
//    }

    if (!fixed_camera){
        DEBUG_MSG("update parent particle weights") ;
        // create key vector for reducing normalizers
        device_vector<int> dev_keys(n_normalizers) ;
        device_vector<int>::iterator it = dev_keys.begin() ;
        for ( int n = 0 ; n < n_particles_ ; n++ ){
            it = thrust::fill_n(it,n_measure,n) ;
        }

        // sum the log-valued normalizers
        device_vector<double> dev_normalizer_sums(n_particles_) ;
        thrust::reduce_by_key(dev_keys.begin(),dev_keys.end(),
                              dev_normalizers.begin(),
                              make_discard_iterator(),
                              dev_normalizer_sums.begin()) ;

//        print_vector(dev_normalizer_sums) ;

        // compute predicted cardinalities
        device_vector<double> dev_cardinalities_predict(n_particles_) ;
        thrust::reduce_by_key(dev_gaussian_indices_.begin(),
                              dev_gaussian_indices_.end(),
                              dev_feature_weights_.begin(),
                              make_discard_iterator(),
                              dev_cardinalities_predict.begin()) ;

        if (verbosity_ >= 3){
            DEBUG_MSG("predicted cardinalities") ;
            print_vector(dev_cardinalities_predict) ;
        }

        // add predicted cardinalities to normalizer sums
        thrust::transform(dev_normalizer_sums.begin(),
                          dev_normalizer_sums.end(),
                          dev_cardinalities_predict.begin(),
                          dev_normalizer_sums.begin(),
                          thrust::plus<double>()) ;

        // exponentiate
        thrust::transform(dev_normalizer_sums.begin(),
                         dev_normalizer_sums.end(),
                         dev_normalizer_sums.begin(),
                         exponentiate<double,double>()) ;

        // copy to host and reweight particles
        host_vector<double> normalizer_sums = dev_normalizer_sums ;

//        print_vector(normalizer_sums) ;

        thrust::transform(particle_weights_.begin(),
                          particle_weights_.end(),
                          normalizer_sums.begin(),
                          particle_weights_.begin(),
                          thrust::multiplies<double>()) ;

        // normalize particle weights
        double sum = thrust::reduce(particle_weights_.begin(),
                                 particle_weights_.end()) ;
        DEBUG_VAL(sum) ;
        thrust::transform(particle_weights_.begin(),
                         particle_weights_.end(),
                          particle_weights_.begin(),
                         divide_by<double,double>(sum)) ;
    }

    // recalculate offsets for updated map size
    host_vector<int> map_offsets = dev_map_offsets ;
    for ( int n = 0 ; n < n_particles_+1 ; n++ ){
        map_offsets[n] *= (n_measure) ;

        map_offsets[n] += n_measure*n ;
    }
    dev_map_offsets = map_offsets ;
//    print_vector(map_offsets) ;

    gaussian_indices_updated_.resize(n_update);
    int k = 0 ;
    for ( int i = 0 ; i < n_update ; i++ ){
        if ( i < map_offsets[k+1] )
            gaussian_indices_updated_[i] = k ;
        else
            gaussian_indices_updated_[i] = ++k ;
    }

    // save updated features for debugging

    if (config_.lookup("save_all_maps")){
        dev_features_updated_.resize(dev_features_update.size());
        thrust::copy(dev_features_update.begin(),
                     dev_features_update.end(),
                     dev_features_updated_.begin()) ;
        DEBUG_VAL(dev_features_updated_.size()) ;
    }

    // ---------------- GM reduction ------------------------------------- //
    device_vector<int> dev_merged_sizes(n_particles_) ;
    device_vector<Gaussian6D> dev_gaussians_merged_tmp(n_update) ;

//    print_vector(dev_merge_flags) ;

    DEBUG_MSG("Performing GM reduction") ;
    phdUpdateMergeKernel<<<n_particles_,256>>>
     (raw_pointer_cast(&dev_features_update[0]),
      raw_pointer_cast(&dev_gaussians_merged_tmp[0]),
      raw_pointer_cast(&dev_merged_sizes[0]),
      raw_pointer_cast(&dev_merge_flags[0]),
      raw_pointer_cast(&dev_map_offsets[0]),
        n_particles_, config_.lookup("min_separation")) ;
    cudaDeviceSynchronize() ;

    DEBUG_MSG("copy results back to host") ;
    host_vector<int> merged_sizes = dev_merged_sizes ;
//    print_vector(merged_sizes) ;
    DEBUG_MSG("thrust::reduce") ;
    int n_merged_total = thrust::reduce(merged_sizes.begin(),
                                        merged_sizes.end()) ;

    dev_features_.resize(n_merged_total);
    device_vector<Gaussian6D>::iterator it_merged = dev_features_.begin() ;

    dev_gaussian_indices_.resize(n_merged_total);
    device_vector<int>::iterator it_idx = dev_gaussian_indices_.begin() ;
    nvtxRangeId_t copy_merged_id = nvtxRangeStartA("Copying merged features") ;
    for ( int n = 0 ; n < merged_sizes.size() ; n++){
        it_merged = thrust::copy_n(&dev_gaussians_merged_tmp[map_offsets[n]],
                        merged_sizes[n],
                        it_merged) ;
        it_idx = thrust::fill_n(it_idx,merged_sizes[n],n) ;
    }
    nvtxRangeEnd(copy_merged_id);

//    print_vector(dev_gaussian_indices_) ;

    DEBUG_VAL(n_merged_total) ;

    if ( verbosity_ >= 3 ){
        DEBUG_MSG("Merged gaussians and merge flags: ") ;
        for (int n = 0 ; n < n_merged_total  ; n++){
            Gaussian6D g = dev_features_[n] ;
            print_feature(g) ;
        }
    }

//    DEBUG_MSG("Replace out of range features and non-detection terms") ;
//    recombineFeatures();
//    n_merged_total = dev_features_.size() ;
//    DEBUG_VAL(n_merged_total) ;

    DEBUG_MSG("get the updated feature weights") ;
//    device_vector<double> dev_merged_weights(n_merged_total) ;
    dev_feature_weights_.resize(n_merged_total);
    thrust::transform(dev_features_.begin(),
                      dev_features_.end(),
                      dev_feature_weights_.begin(),
                      get_weight()) ;


    // ---- Transform features back to Euclidean space ------------------------

    DEBUG_MSG("sample disparity space gaussians") ;
    int n_feature_particles = particles_per_feature_*n_merged_total ;
    dev_u_.resize(n_feature_particles);
    dev_v_.resize(n_feature_particles);
    dev_d_.resize(n_feature_particles);
    dev_vu_.resize(n_feature_particles);
    dev_vv_.resize(n_feature_particles);
    dev_vd_.resize(n_feature_particles);

    time_t time_val ;
    time(&time_val) ;
    nvtxRangeId_t sample_range_id = nvtxRangeStartA("Sampling disparity gaussians") ;
    thrust::for_each(make_zip_iterator(make_tuple(
                                           make_counting_iterator(0),
                                           dev_u_.begin(),
                                           dev_v_.begin(),
                                           dev_d_.begin(),
                                           dev_vu_.begin(),
                                           dev_vv_.begin(),
                                           dev_vd_.begin())),
                     make_zip_iterator(make_tuple(
                                           make_counting_iterator(n_feature_particles),
                                           dev_u_.end(),
                                           dev_v_.end(),
                                           dev_d_.end(),
                                           dev_vu_.end(),
                                           dev_vv_.end(),
                                           dev_vd_.end())),
                     sample_disparity_gaussian(
                        raw_pointer_cast(&dev_features_[0]),
                     particles_per_feature_, int(time_val)
                     )
                ) ;
    nvtxRangeEnd(sample_range_id);

    // create uniform particle weights for sampled gaussians
    dev_particle_weights_.resize(n_feature_particles);
    thrust::fill(dev_particle_weights_.begin(), dev_particle_weights_.end(),
                 1.0/particles_per_feature_) ;

    if ( verbosity_ >= 3 ){
        cout << "disparity particles: " << endl ;
        for ( int i = 0 ; i < n_feature_particles ; i++){
            cout << dev_u_[i] << " " << dev_v_[i] << " " << dev_d_[i] << " "
                 << dev_vu_[i] << " " << dev_vv_[i] << " " << dev_vd_[i] << endl ;
        }
    }

//    // check for nan values
//    bool nan_values = thrust::any_of(dev_vd_.begin(),dev_vd_.end(),check_nan<double>()) ;
//    if (nan_values and verbosity_ >= 1){
//        DEBUG_MSG("FOUND NAN PARTICLES!") ;

//        DEBUG_MSG("Merged gaussians and merge flags: ") ;
//        for (int n = 0 ; n < n_merged_total  ; n++){
//            Gaussian6D g = dev_features_[n] ;
//            print_feature(g) ;
//        }

//        cout << "disparity particles: " << endl ;
//        for ( int i = 0 ; i < n_feature_particles ; i++){
//            cout << dev_u_[i] << " " << dev_v_[i] << " " << dev_d_[i] << " "
//                 << dev_vu_[i] << " " << dev_vv_[i] << " " << dev_vd_[i] << endl ;
//        }
//    }

    // create particle indices for updated features
    device_vector<int> dev_map_sizes(n_particles_) ;
    thrust::reduce_by_key(
        dev_gaussian_indices_.begin(),
        dev_gaussian_indices_.end(),
        make_constant_iterator(1),
        make_discard_iterator(),
        dev_map_sizes.begin()) ;
    host_vector<int> map_sizes = dev_map_sizes ;

    dev_particle_indices_.resize(n_feature_particles);
    device_vector<int>::iterator it_features = dev_particle_indices_.begin() ;
    for ( int n = 0 ; n < n_particles_ ; n++ ){
        it_features = thrust::fill_n(
                    it_features,
                    map_sizes[n]*particles_per_feature_,
                    n) ;
    }

    DEBUG_MSG("recombine with nondetection terms") ;
    recombineNonDetections() ;
    n_feature_particles = dev_particle_indices_.size() ;

//    // compute gaussian indices from combined particle indices
//    int n_feature_gaussians = n_feature_particles/particles_per_feature_ ;
//    dev_gaussian_indices_.resize(n_feature_gaussians) ;
//    thrust::gather(make_transform_iterator(make_counting_iterator(0),
//                                           times(particles_per_feature_)),
//                   make_transform_iterator(make_counting_iterator(n_feature_gaussians),
//                                           times(particles_per_feature_)),
//                   dev_particle_indices_.begin(),
//                   dev_gaussian_indices_.begin()) ;


    DEBUG_MSG("transform to particles to euclidean space") ;
    computeEuclideanParticles(fixed_camera) ;

//    DEBUG_MSG("particle weights") ;
//    print_vector(dev_particle_weights_) ;

//    cout << "euclidean particles: " << endl ;
//    for ( int i = 0 ; i < n_feature_particles ; i++){
//        cout << dev_x_[i] << " " << dev_y_[i] << " " << dev_z_[i] << " "
//             << dev_vx_[i] << " " << dev_vy_[i] << " " << dev_vz_[i] << endl ;
//    }

//    print_vector(dev_particle_indices_) ;

    // refit gaussians so that we save the non-detection terms in the posterior
    // map
    if (config_.lookup("save_all_maps")){
        fitGaussians();
    }

//    DEBUG_MSG("gaussian indices") ;
//    print_vector(dev_gaussian_indices_) ;
}


/// write gaussian mixtures to matvar. this version only writes the mixture
/// corresponding to idx
matvar_t* SCPHDCameraCalibration::writeGaussianMixtureMatVar(host_vector<Gaussian6D> gm,
                                                             host_vector<int> indices,
                                                            const char* varname,
                                                             int idx){
    const char* fields[3] = {"weights","means","covs"} ;
    size_t dims[2] ;
    dims[0] = 1 ;
    dims[1] = 1 ;
    matvar_t* matvar = Mat_VarCreateStruct(varname,2,dims,fields,3) ;

    int n_features = thrust::count(indices.begin(),indices.end(),idx) ;
    host_vector<Gaussian6D> gm_i(n_features) ;
    thrust::copy_if(gm.begin(),gm.end(),indices.begin(),gm_i.begin(),
                    equals<int>(idx)) ;
    host_vector<double> weights_vec(n_features) ;
    host_vector<double> means_vec(n_features*6) ;
    host_vector<double> covs_vec(n_features*6*6) ;

    host_vector<double>::iterator means_it = means_vec.begin() ;
    host_vector<double>::iterator covs_it = covs_vec.begin() ;


    for ( int n = 0 ; n < n_features ; n++ ){
        weights_vec[n] = gm_i[n].weight ;

        means_it = thrust::copy_n(&(gm_i[n] .mean[0]),6,means_it) ;
        covs_it = thrust::copy_n(&(gm_i[n].cov[0]),36,covs_it) ;
    }

    dims[0] = 1 ;
    dims[1] = n_features ;
    matvar_t* w = Mat_VarCreate(NULL,MAT_C_DOUBLE,MAT_T_DOUBLE,2,dims,weights_vec.data(),0) ;

    dims[0] = 6 ;
    matvar_t* m = Mat_VarCreate(NULL,MAT_C_DOUBLE,MAT_T_DOUBLE,2,dims,means_vec.data(),0) ;

    size_t cov_dims[3] ;
    cov_dims[0] = 6 ;
    cov_dims[1] = 6 ;
    cov_dims[2] = n_features ;
    matvar_t* c = Mat_VarCreate(NULL,MAT_C_DOUBLE,MAT_T_DOUBLE,3,cov_dims,covs_vec.data(),0) ;

    Mat_VarSetStructFieldByName(matvar,"weights",0,w) ;
    Mat_VarSetStructFieldByName(matvar,"means",0,m) ;
    Mat_VarSetStructFieldByName(matvar,"covs",0,c) ;

    return matvar ;
}

/// write gaussian mixtures to matvar
matvar_t* SCPHDCameraCalibration::writeGaussianMixtureMatVar(host_vector<Gaussian6D> gm,
                                                             host_vector<int> indices,
                                                            const char* varname){
    const char* fields[3] = {"weights","means","covs"} ;
    int n_maps = *thrust::max_element(indices.begin(),indices.end())+1 ;
    size_t dims[2] ;
    dims[0] = n_maps ;
    dims[1] = 1 ;
    matvar_t* matvar = Mat_VarCreateStruct(varname,2,dims,fields,3) ;

    for ( int i = 0 ; i < n_maps ; i++ ){
        int n_features = thrust::count(indices.begin(),indices.end(),i) ;
        host_vector<Gaussian6D> gm_i(n_features) ;
        thrust::copy_if(gm.begin(),gm.end(),indices.begin(),gm_i.begin(),
                        equals<int>(i)) ;
        host_vector<double> weights_vec(n_features) ;
        host_vector<double> means_vec(n_features*6) ;
        host_vector<double> covs_vec(n_features*6*6) ;

        host_vector<double>::iterator means_it = means_vec.begin() ;
        host_vector<double>::iterator covs_it = covs_vec.begin() ;


        for ( int n = 0 ; n < n_features ; n++ ){
            weights_vec[n] = gm_i[n].weight ;

            means_it = thrust::copy_n(&(gm_i[n] .mean[0]),6,means_it) ;
            covs_it = thrust::copy_n(&(gm_i[n].cov[0]),36,covs_it) ;
        }

        dims[0] = 1 ;
        dims[1] = n_features ;
        matvar_t* w = Mat_VarCreate(NULL,MAT_C_DOUBLE,MAT_T_DOUBLE,2,dims,weights_vec.data(),0) ;

        dims[0] = 6 ;
        matvar_t* m = Mat_VarCreate(NULL,MAT_C_DOUBLE,MAT_T_DOUBLE,2,dims,means_vec.data(),0) ;

        size_t cov_dims[3] ;
        cov_dims[0] = 6 ;
        cov_dims[1] = 6 ;
        cov_dims[2] = n_features ;
        matvar_t* c = Mat_VarCreate(NULL,MAT_C_DOUBLE,MAT_T_DOUBLE,3,cov_dims,covs_vec.data(),0) ;

        Mat_VarSetStructFieldByName(matvar,"weights",i,w) ;
        Mat_VarSetStructFieldByName(matvar,"means",i,m) ;
        Mat_VarSetStructFieldByName(matvar,"covs",i,c) ;
    }

    return matvar ;
}

void SCPHDCameraCalibration::writeMat(const char *filename)
{
    mat_t* matfp = Mat_CreateVer(filename,NULL,MAT_FT_MAT73) ;

    // pack particle states into array
    host_vector<Extrinsics> h_particle_states = dev_particle_states_ ;
    vector<double> particles_array ;
    for ( int n = 0 ; n < n_particles_ ; n++){
        particles_array.push_back(h_particle_states[n].cartesian.x);
        particles_array.push_back(h_particle_states[n].cartesian.y);
        particles_array.push_back(h_particle_states[n].cartesian.z);
        particles_array.push_back(h_particle_states[n].cartesian.vz);
        particles_array.push_back(h_particle_states[n].cartesian.vy);
        particles_array.push_back(h_particle_states[n].cartesian.vz);
        particles_array.push_back(h_particle_states[n].angular.x);
        particles_array.push_back(h_particle_states[n].angular.y);
        particles_array.push_back(h_particle_states[n].angular.z);
        particles_array.push_back(h_particle_states[n].angular.vz);
        particles_array.push_back(h_particle_states[n].angular.vy);
        particles_array.push_back(h_particle_states[n].angular.vz);
    }
    size_t dims[2] ;
    dims[0] = 12 ;
    dims[1] = n_particles_ ;
//    cout << particles_array[13*12] << endl ;
    matvar_t* matvar = Mat_VarCreate("particles",MAT_C_SINGLE,MAT_T_DOUBLE,
                                     2,dims,particles_array.data(),0) ;
    Mat_VarWrite(matfp,matvar,MAT_COMPRESSION_NONE) ;
    Mat_VarFree(matvar) ;

    DEBUG_MSG("write particle weights") ;
    dims[0] = n_particles_ ;
    dims[1] = 1 ;
    matvar_t* w = Mat_VarCreate("weights",MAT_C_DOUBLE,MAT_T_DOUBLE,2,
                                dims,particle_weights_.data(),0) ;
    Mat_VarWrite(matfp,w,MAT_COMPRESSION_NONE) ;
    Mat_VarFree(w) ;

    DEBUG_MSG("write feature particles as array of structures") ;
    size_t struct_dims[2] ;
    struct_dims[0] = 1 ;
    struct_dims[1] = 1 ;
    const char* fieldnames[2] = {"weights","particles"} ;
    matvar_t* features = Mat_VarCreateStruct("features",2,struct_dims,fieldnames,2) ;
    host_vector<double> x = dev_x_ ;
    host_vector<double> y = dev_y_ ;
    host_vector<double> z = dev_z_ ;
    host_vector<double> vx = dev_vx_ ;
    host_vector<double> vy = dev_vy_ ;
    host_vector<double> vz = dev_vz_ ;

    // find maximum particle for MAP map estimate
    thrust::host_vector<double>::iterator max_ptr = thrust::max_element(
                particle_weights_.begin(),particle_weights_.end()) ;
    int max_idx = max_ptr - particle_weights_.begin() ;
    DEBUG_VAL(max_idx) ;

    if (x.size() > 0){
        host_vector<double> feature_weights = dev_feature_weights_ ;
        host_vector<int> particle_indices = dev_particle_indices_ ;

        int i = 0 ;
        while(particle_indices[i] != max_idx)
            i++ ;

        int offset = i ;
        particles_array.clear();
        while (particle_indices[i] == max_idx){
            particles_array.push_back(x[i]);
            particles_array.push_back(y[i]);
            particles_array.push_back(z[i]);
            particles_array.push_back(vx[i]);
            particles_array.push_back(vy[i]);
            particles_array.push_back(vz[i]);
            i += 1 ;
        }

        int n_features = particles_array.size()/6/particles_per_feature_ ;
        DEBUG_VAL(n_features) ;
        dims[0] = n_features ;
        dims[1] = 1 ;
        int weight_idx = offset/particles_per_feature_ ;
        DEBUG_VAL(weight_idx) ;
        matvar_t* weights = Mat_VarCreate(NULL,MAT_C_DOUBLE,MAT_T_DOUBLE,
                                          2,dims,&feature_weights[weight_idx],0) ;
//        offset += n_features ;

        dims[0] = 6 ;
        dims[1] = n_features*particles_per_feature_ ;
        matvar_t* particles = Mat_VarCreate(NULL,MAT_C_DOUBLE,MAT_T_DOUBLE,
                                            2,dims,particles_array.data(),0) ;

        Mat_VarSetStructFieldByName(features,"weights",0,weights) ;
        Mat_VarSetStructFieldByName(features,"particles",0,particles) ;
    }
    else
    {
        DEBUG_MSG("All maps are empty") ;
        for ( int n = 0 ; n < n_particles_ ; n++){
            dims[0] = 0 ;
            dims[1] = 0 ;
            double dummy = 0 ;
            matvar_t* empty1 = Mat_VarCreate(NULL,MAT_C_EMPTY,
                                           MAT_T_DOUBLE,2,dims,&dummy,0) ;
            matvar_t* empty2 = Mat_VarCreate(NULL,MAT_C_EMPTY,
                                           MAT_T_DOUBLE,2,dims,&dummy,0) ;
            Mat_VarSetStructFieldByName(features,"weights",n,empty1) ;
            Mat_VarSetStructFieldByName(features,"particles",n,empty2) ;
        }
    }
    Mat_VarWrite(matfp,features,MAT_COMPRESSION_NONE) ;
    Mat_VarFree(features) ;

    if (config_.lookup("save_all_maps")){
        DEBUG_MSG("write_predicted GM") ;
        matvar_t* predicted = NULL ;
        if ( dev_features_predicted_.size() > 0 ){
            host_vector<Gaussian6D> gm = dev_features_predicted_ ;
            predicted = writeGaussianMixtureMatVar(gm,gaussian_indices_predicted_,
                                                   "features_predicted") ;
        }
        else{
            DEBUG_MSG("no predicted features") ;
            double dummy = 0 ;
            dims[0] = 0 ;
            dims[1] = 0 ;
            predicted = Mat_VarCreate("features_predicted",MAT_C_EMPTY,
                                    MAT_T_DOUBLE,2,dims,&dummy,0) ;
        }
        Mat_VarWrite(matfp,predicted,MAT_COMPRESSION_NONE) ;
        Mat_VarFree(predicted) ;

        DEBUG_MSG("write updated GM") ;
        matvar_t* updated = NULL ;
        if (dev_features_updated_.size() > 0){
            host_vector<Gaussian6D> gm = dev_features_updated_ ;
            updated = writeGaussianMixtureMatVar(gm,
                                                 gaussian_indices_updated_,
                                                 "features_updated");
        }
        else{
            DEBUG_MSG("no updated features") ;
            double dummy = 0 ;
            dims[0] = 0 ;
            dims[1] = 0 ;
            updated = Mat_VarCreate("features_updated",MAT_C_EMPTY,
                                    MAT_T_DOUBLE,2,dims,&dummy,0) ;
        }
        Mat_VarWrite(matfp,updated,MAT_COMPRESSION_NONE) ;
        Mat_VarFree(updated) ;

        DEBUG_MSG("write merged GM") ;
        matvar_t* merged = NULL ;
        if (dev_features_.size() > 0){
            host_vector<Gaussian6D> gm = dev_features_ ;
            host_vector<int> idx = dev_gaussian_indices_ ;
            merged = writeGaussianMixtureMatVar(gm,idx,"features_merged");
        }
        else{
            DEBUG_MSG("no merged features") ;
            double dummy = 0 ;
            dims[0] = 0 ;
            dims[1] = 0 ;
            merged = Mat_VarCreate("features_merged",MAT_C_EMPTY,
                                    MAT_T_DOUBLE,2,dims,&dummy,0) ;
        }
        Mat_VarWrite(matfp,merged,MAT_COMPRESSION_NONE) ;
        Mat_VarFree(merged) ;
    }


    DEBUG_MSG("Close matfile") ;
    Mat_Close(matfp) ;
}

void SCPHDCameraCalibration::fitGaussians()
{
    int n_features_total = dev_x_.size()/particles_per_feature_ ;
    int n_blocks = n_features_total ;
    dev_features_.resize(n_features_total);


//    DEBUG_MSG("particle pd:") ;
//    print_vector(dev_particle_pd_) ;

    DEBUG_MSG("normalize weights") ;
    normalizeWeightsKernel<<<n_blocks,256>>>(
            raw_pointer_cast(&dev_particle_weights_[0]),
            particles_per_feature_);
    cudaDeviceSynchronize() ;

//    DEBUG_MSG("particle weights:") ;
//    print_vector(dev_particle_weights_) ;

    n_blocks = min(cuda_dev_props_.maxGridSize[0],n_features_total) ;

    DEBUG_MSG("fitGaussiansKernel") ;
    fitGaussiansKernel<<<n_blocks,256>>>
           (raw_pointer_cast(&dev_u_[0]),
            raw_pointer_cast(&dev_v_[0]),
            raw_pointer_cast(&dev_d_[0]),
            raw_pointer_cast(&dev_vu_[0]),
            raw_pointer_cast(&dev_vv_[0]),
            raw_pointer_cast(&dev_vd_[0]),
            raw_pointer_cast(&dev_particle_weights_[0]),
            raw_pointer_cast(&dev_feature_weights_[0]),
            n_features_total,
            raw_pointer_cast(&dev_features_[0]),
            particles_per_feature_) ;
    cudaDeviceSynchronize() ;


//    cout << "fitted gaussians: " << endl ;
//    for  ( int i = 0 ;i < dev_features_.size() ; i++){
//        Gaussian6D g = dev_features_[i] ;
//        print_feature(g) ;
//    }

//    cout << "particle indices: " << endl ;
//    for ( int i = 0 ; i < dev_particle_indices_.size() ; i++){
//        cout << dev_particle_indices_[i] << " " ;
//    }
//    cout << endl ;



//    cout << "gaussian indices: " << endl ;
//    for ( int i = 0 ; i < dev_gaussian_indices_.size() ; i++){
//        cout << dev_gaussian_indices_[i] << " " ;
//    }
//    cout << endl ;

}

void SCPHDCameraCalibration::separateOutOfRange()
{
    bool out_of_range = thrust::any_of(dev_pd_.begin(),dev_pd_.end(),equals<double>(0.0)) ;

    if (out_of_range){
        // partition the features and indices based on pd
        device_vector<Gaussian6D>::iterator middle_features =
                thrust::stable_partition(dev_features_.begin(),dev_features_.end(),
                                 dev_pd_.begin(),thrust::identity<double>()) ;

        device_vector<int>::iterator middle_idx =
                thrust::stable_partition(dev_gaussian_indices_.begin(),
                                         dev_gaussian_indices_.end(),
                                         dev_pd_.begin(),thrust::identity<double>()) ;


        // copy out of range features and indices
        size_t n_out = dev_gaussian_indices_.end() - middle_idx ;
        dev_features_out_of_range_.resize(n_out);
        dev_indices_out_of_range_.resize(n_out);

        DEBUG_VAL(n_out) ;

        thrust::copy(middle_features,dev_features_.end(),
                     dev_features_out_of_range_.begin()) ;
        thrust::copy(middle_idx,dev_gaussian_indices_.end(),
                     dev_indices_out_of_range_.begin()) ;

        // shrink in range arrays
        size_t n_in = middle_idx - dev_gaussian_indices_.begin() ;

        dev_features_.resize(n_in);
        dev_features_.shrink_to_fit();

        dev_gaussian_indices_.resize(n_in);
        dev_gaussian_indices_.shrink_to_fit();
    }
    else{
        dev_features_out_of_range_.resize(0);
        dev_indices_out_of_range_.resize(0);
    }

    return ;
}

void SCPHDCameraCalibration::recombineFeatures()
{
    int n_out = dev_features_out_of_range_.size() ;
    int n_in = dev_features_.size() ;

    DEBUG_VAL(n_in) ;
    DEBUG_VAL(n_out) ;

    if (n_out == 0)
        return ;

    device_vector<Gaussian6D> dev_features_combined(n_in+n_out) ;
    device_vector<int> dev_indices_combined(n_in+n_out) ;

//    device_vector<Gaussian6D>::iterator it_features = dev_features_combined.begin() ;
//    device_vector<int>::iterator it_idx = dev_indices_combined.begin() ;

//    nvtxRangeId_t id = nvtxRangeStartA("recombineFeatures for loop") ;
//    for ( int n = 0 ; n < n_particles_ ; n++ ){
//        it_features = thrust::copy_if(dev_features_.begin(), dev_features_.end(),
//                                    dev_gaussian_indices_.begin(),it_features,
//                                    equals<int>(n)) ;
//        it_features = thrust::copy_if(dev_features_out_of_range_.begin(),
//                                      dev_features_out_of_range_.end(),
//                                      dev_indices_out_of_range_.begin(),
//                                      it_features, equals<int>(n)) ;

//        it_idx = thrust::copy_if(dev_gaussian_indices_.begin(),
//                                 dev_gaussian_indices_.end(),
//                                 it_idx, equals<int>(n)) ;
//        it_idx = thrust::copy_if(dev_indices_out_of_range_.begin(),
//                                 dev_indices_out_of_range_.end(),
//                                 it_idx, equals<int>(n)) ;
//    }
//    nvtxRangeEnd(id) ;

    nvtxRangeId_t id = nvtxRangeStartA("Compute indices for recombineFeaturesKernel") ;
//    device_vector<int> d_map_sizes_out(n_particles_) ;
//    device_vector<int> d_map_sizes(n_particles_) ;

//    device_vector<int> d_map_offsets_out(n_particles_+1,0) ;
//    device_vector<int> d_map_offsets(n_particles_+1,0) ;

//    thrust::reduce_by_key(dev_indices_out_of_range_.begin(),
//                          dev_indices_out_of_range_.end(),
//                          make_constant_iterator(1),
//                          make_discard_iterator(),
//                          d_map_sizes_out.begin()) ;

//    thrust::reduce_by_key(dev_gaussian_indices_.begin(),
//                          dev_gaussian_indices_.end(),
//                          make_constant_iterator(1),
//                          make_discard_iterator(),
//                          d_map_sizes.begin()) ;

//    thrust::inclusive_scan(d_map_sizes_out.begin(),
//                           d_map_sizes_out.end(),
//                           d_map_offsets_out.begin()+1) ;
//    thrust::inclusive_scan(d_map_sizes.begin(),
//                           d_map_sizes.end(),
//                           d_map_offsets.begin()+1) ;

    device_vector<int> d_map_offsets = computeMapOffsets(dev_gaussian_indices_) ;
    device_vector<int> d_map_offsets_out = computeMapOffsets(dev_indices_out_of_range_) ;

    nvtxRangeEnd(id);

    int n_threads = 128 ;
    int n_blocks = ceil(double(n_in+n_out)/n_threads) ;
    interleaveKernel<<<n_blocks,n_threads>>>(
            raw_pointer_cast(&dev_features_[0]),
            raw_pointer_cast(&dev_features_out_of_range_[0]),
            raw_pointer_cast(&d_map_offsets[0]),
            raw_pointer_cast(&d_map_offsets_out[0]),
            n_particles_, n_in+n_out,
            raw_pointer_cast(&dev_features_combined[0]),
            raw_pointer_cast(&dev_indices_combined[0])
            );

    dev_features_ = dev_features_combined ;
    dev_gaussian_indices_ = dev_indices_combined ;
}

device_vector<int> SCPHDCameraCalibration::computeMapOffsets(device_vector<int> indices)
{
    host_vector<int> map_sizes ;
    if (indices.size() > 0){
        device_vector<int> dev_map_sizes(n_particles_) ;
        thrust::reduce_by_key(indices.begin(),
                              indices.end(),
                              make_constant_iterator(1),
                              make_discard_iterator(),
                              dev_map_sizes.begin()) ;
        map_sizes = dev_map_sizes ;
    }
    else{
        map_sizes.resize(n_particles_,0);
    }

    // compute the indexing offsets with an inclusive scan
    host_vector<int> map_offsets(n_particles_+ 1) ;
    map_offsets[0] = 0 ;
    thrust::inclusive_scan(map_sizes.begin(),map_sizes.end(),
                           map_offsets.begin()+1) ;
    device_vector<int> dev_map_offsets = map_offsets ;
    return dev_map_offsets ;
}



thrust::device_vector<Gaussian6D> SCPHDCameraCalibration::computeBirths(vector<double> u, vector<double> v)
{
    nvtxRangeId_t birth_id = nvtxRangeStartA("compute births") ;
    int n_measure = u.size() ;
    host_vector<Gaussian6D> h_births(n_measure) ;

    DisparityPoint birth_mean ;
    const Setting& birth_config = config_.lookup("birth") ;
    birth_mean.d = birth_config["d0"] ;
    birth_mean.vu = birth_config["vu0"] ;
    birth_mean.vv = birth_config["vv0"] ;
    birth_mean.vd = birth_config["vd0"] ;

    DisparityPoint birth_vars ;
    birth_vars.u = measurement_model_.std_u()*measurement_model_.std_u() ;
    birth_vars.v = measurement_model_.std_v()*measurement_model_.std_v() ;
    birth_vars.d = birth_config["var_d0"] ;
    birth_vars.vu = birth_config["var_vu0"] ;
    birth_vars.vv = birth_config["var_vv0"] ;
    birth_vars.vd = birth_config["var_vd0"] ;

    double birth_weight = birth_config["w0"] ;

    for ( int m = 0 ; m < n_measure ; m++ ){
        Gaussian6D f ;
        f.mean[0] = u[m] ;
        f.mean[1] = v[m] ;
        f.mean[2] = birth_mean.d ;
        f.mean[3] = birth_mean.vu ;
        f.mean[4] = birth_mean.vv ;
        f.mean[5] = birth_mean.vd ;

        for ( int i = 0 ; i < 36 ; i++)
            f.cov[i] = 0 ;
        f.cov[0] = birth_vars.u ;
        f.cov[7] = birth_vars.v ;
        f.cov[14] = birth_vars.d ;
        f.cov[21] = birth_vars.vu ;
        f.cov[28] = birth_vars.vv ;
        f.cov[35] = birth_vars.vd ;

        f.weight = safeLog(birth_weight) ;

        h_births[m] = f ;
    }

    device_vector<Gaussian6D> d_births = h_births ;
    nvtxRangeEnd(birth_id);
    return d_births ;
}


double SCPHDCameraCalibration::computeNeff(){
//    print_vector(particle_weights_) ;
    double sum_of_squares = thrust::transform_reduce(particle_weights_.begin(),
                                                    particle_weights_.end(),
                                                     square<double>(),0.0,
                                                     thrust::plus<double>()) ;
//    DEBUG_VAL(sum_of_squares) ;
    return 1.0/(n_particles_*sum_of_squares) ;
}

void SCPHDCameraCalibration::resample()
{
    double n_eff = computeNeff() ;
    DEBUG_VAL(n_eff) ;
    double min_neff = config_.lookup("min_neff") ;
    if (n_eff >= min_neff)
        return ;

//    print_vector(particle_weights_) ;
    host_vector<int> idx_resample(n_particles_) ;
    double interval = 1.0/n_particles_ ;

    time_t time_val ;
    time(&time_val) ;
    thrust::default_random_engine rng(time_val) ;
    thrust::uniform_real_distribution<double> u01(0,interval) ;
    double r = u01(rng) ;
    double c = particle_weights_[0] ;
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
            c += particle_weights_[i] ;
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
    DEBUG_MSG("thrust::exclusive_scan") ;
    thrust::exclusive_scan(dev_map_sizes.begin(), dev_map_sizes.end(),
                           dev_map_offsets.begin()) ;
    map_sizes = dev_map_sizes ;
    map_offsets = dev_map_offsets ;

//    print_vector(map_sizes) ;
//    print_vector(map_offsets) ;

    int n_particles_resample = 0 ;
    for ( int n = 0 ; n < n_particles_ ; n++){
        n_particles_resample += map_sizes[idx_resample[n]] ;
    }

    device_vector<double> dev_new_x(n_particles_resample) ;
    device_vector<double> dev_new_y(n_particles_resample) ;
    device_vector<double> dev_new_z(n_particles_resample) ;
    device_vector<double> dev_new_vx(n_particles_resample) ;
    device_vector<double> dev_new_vy(n_particles_resample) ;
    device_vector<double> dev_new_vz(n_particles_resample) ;

    device_vector<double> dev_new_particle_weights(n_particles_resample) ;
    device_vector<double> dev_new_feature_weights(n_particles_resample/particles_per_feature_) ;

    device_vector<double>::iterator it_x = dev_new_x.begin() ;
    device_vector<double>::iterator it_y = dev_new_y.begin() ;
    device_vector<double>::iterator it_z = dev_new_z.begin() ;
    device_vector<double>::iterator it_vx = dev_new_vx.begin() ;
    device_vector<double>::iterator it_vy = dev_new_vy.begin() ;
    device_vector<double>::iterator it_vz = dev_new_vz.begin() ;
    device_vector<double>::iterator it_weight = dev_new_particle_weights.begin() ;
    device_vector<double>::iterator it_weight_g = dev_new_feature_weights.begin() ;

    dev_particle_indices_.resize(n_particles_resample);
    device_vector<int>::iterator it_idx = dev_particle_indices_.begin() ;

    dev_gaussian_indices_.resize(n_particles_resample/particles_per_feature_);
    device_vector<int>::iterator it_idx_g = dev_gaussian_indices_.begin() ;

    if (verbosity_ >= 1){
        DEBUG_MSG("resample indices:") ;
        for ( int i = 0 ; i < n_particles_ ; i++){
            cout << idx_resample[i] << " " ;
            if ( (i % 20) == 0 && i > 0 ){
                cout << endl ;
            }
        }
    }

    cout << endl ;
    DEBUG_MSG("thrust::copy_n") ;
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
        it_weight = thrust::copy_n(dev_particle_weights_.begin()+offset,
                                   map_sizes[idx],it_weight) ;
        it_weight_g = thrust::copy_n(dev_feature_weights_.begin()+offset/particles_per_feature_,
                                   map_sizes[idx]/particles_per_feature_,
                                    it_weight_g) ;


        it_idx = thrust::fill_n(it_idx,map_sizes[idx],i) ;
        it_idx_g = thrust::fill_n(it_idx_g,
                                  map_sizes[idx]/particles_per_feature_,
                                  i) ;
    }

    // save resampled values
    DEBUG_MSG("copy vectors") ;
    dev_particle_states_ = dev_new_particles ;
    dev_x_ = dev_new_x ;
    dev_y_ = dev_new_y ;
    dev_z_ = dev_new_z ;
    dev_vx_ = dev_new_vx ;
    dev_vy_ = dev_new_vy ;
    dev_vz_ = dev_new_vz ;
    dev_particle_weights_ = dev_new_particle_weights ;
    dev_feature_weights_ = dev_new_feature_weights ;

    // reset particle weights
    DEBUG_MSG("thrust::fill") ;
    thrust::fill(particle_weights_.begin(),particle_weights_.end(),
                 1.0/n_particles_) ;
}





DisparityMeasurementModel SCPHDCameraCalibration::measurement_model() const
{
    return measurement_model_;
}

void SCPHDCameraCalibration::setMeasurement_model(const DisparityMeasurementModel &measurement_model)
{
    measurement_model_ = measurement_model;
}
