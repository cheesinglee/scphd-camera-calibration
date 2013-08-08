//#include <thrust/tuple.h>
//#include <thrust/transform.h>
//#include <thrust/iterator/counting_iterator.h>
//#include <thrust/iterator/constant_iterator.h>
//#include <thrust/iterator/discard_iterator.h>
//#include <thrust/iterator/zip_iterator.h>
//#include <thrust/for_each.h>
//#include <thrust/scan.h>
//#include <thrust/random.h>
//#include <thrust/random/normal_distribution.h>
//#include <thrust/reduce.h>
//#include <thrust/gather.h>
//#include <thrust/fill.h>
//#include <thrust/copy.h>
//#include <thrust/count.h>

//#include "disparitymeasurementmodel.cuh"
//#include "OrientedLinearCVMotionModel3D.cuh"
//#include "device_math.cuh"

//using namespace thrust ;

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

//struct generate_gaussian_noise : public thrust::unary_function<unsigned int, double>
//{
//    thrust::random::normal_distribution<double> dist ;
//    double seed_ ;

//    generate_gaussian_noise(double seed, double mean = 0.0, double std = 1.0){
//        dist = thrust::random::normal_distribution<double>(mean,std) ;
//        seed_ = seed ;
//    }

//    __device__ __host__ double
//    operator()(unsigned int tid){
//        thrust::default_random_engine rng(seed_) ;
//        rng.discard(tid);

//        return dist(rng) ;
//    }
//} ;

//struct generate_uniform_random : public thrust::unary_function<unsigned int, double>
//{
//    thrust::random::uniform_real_distribution<double> dist ;
//    double seed_ ;

//    generate_uniform_random(double seed, double a = 0.0, double b = 1.0){
//        dist = thrust::random::uniform_real_distribution<double>(a,b) ;
//        seed_ = seed ;
//    }

//    __device__ __host__ double
//    operator()(unsigned int tid){
//        thrust::random::default_random_engine rng(seed_) ;
//        rng.discard(tid);

//        return dist(rng) ;
//    }
//} ;

//struct predict_camera{
//    OrientedLinearCVMotionModel3D model_ ;
//    double dt ;
//    predict_camera(OrientedLinearCVMotionModel3D m, double dt)
//        : model_(m), dt(dt) {}

//    template <typename T>
//    __device__ __host__ void
//    operator()(T t){
//        Extrinsics state = get<0>(t) ;
//        double ax = get<1>(t) ;
//        double ay = get<2>(t) ;
//        double az = get<3>(t) ;
//        double ax_a = get<4>(t) ;
//        double ay_a = get<5>(t) ;
//        double az_a = get<6>(t) ;


//        model_.computeNoisyMotion(state.cartesian,state.angular,
//                                  dt,ax,ay,az,ax_a,ay_a,az_a);

//        get<0>(t) = state ;
//    }
//};

//struct predict_features{
//    const double dt_ ;
//    predict_features(double dt) : dt_(dt) {}

//    template <typename Tuple>
//    __device__ void
//    operator()(Tuple t){
//        double x = get<0>(t) ;
//        double y = get<1>(t) ;
//        double z = get<2>(t) ;
//        double vx = get<3>(t) ;
//        double vy = get<4>(t) ;
//        double vz = get<5>(t) ;
//        double ax = get<6>(t) ;
//        double ay = get<7>(t) ;
//        double az = get<8>(t) ;

//        get<0>(t) = x + vx*dt_ + 0.5*ax*dt_*dt_ ;
//        get<1>(t) = y + vy*dt_ + 0.5*ay*dt_*dt_ ;
//        get<2>(t) = z + vz*dt_ + 0.5*az*dt_*dt_ ;
//        get<3>(t) = vx + ax*dt_ ;
//        get<4>(t) = vy + ay*dt_ ;
//        get<5>(t) = vz + az*dt_ ;
//    }
//};

////struct world_to_disparity_transform{
////    DisparityMeasurementModel model_ ;
////    Extrinsics* extrinsics ;

////    world_to_disparity_transform(DisparityMeasurementModel model,
////                                 Extrinsics* e) :
////        model_(model), extrinsics(e) {}

////    template <typename Tuple>
////    __device__ __host__ tuple<double,double,double,double,double,double,bool>
////    operator()(Tuple t){
////        EuclideanPoint p_world ;
////        p_world.x = get<0>(t) ;
////        p_world.y = get<1>(t) ;
////        p_world.z = get<2>(t) ;
////        p_world.vx = get<3>(t) ;
////        p_world.vy = get<4>(t) ;
////        p_world.vz = get<5>(t) ;
////        int idx = get<6>(t) ;
////        Extrinsics e = extrinsics[idx] ;
////        DisparityPoint p_disparity = model_.computeMeasurement(p_world,e) ;

////        bool in_range = ( p_disparity.u >= 0 ) &&
////                ( p_disparity.u <= model_.img_width() ) &&
////                ( p_disparity.v >= 0 ) &&
////                ( p_disparity.v <= model_.img_height() ) &&
////                ( p_disparity.d >= 0 ) ;
////        return make_tuple(p_disparity.u,
////                        p_disparity.v,
////                        p_disparity.d,
////                        p_disparity.vu,
////                        p_disparity.vv,
////                        p_disparity.vd,
////                          in_range) ;
////    }
////};

////struct disparity_to_world_transform{
////    DisparityMeasurementModel model_ ;
////    Extrinsics* extrinsics ;
////    double max_v ;

////    disparity_to_world_transform(DisparityMeasurementModel model,
////                                 Extrinsics* e, double max_v) :
////        model_(model), extrinsics(e), max_v(max_v) {}

////    template <typename Tuple>
////    __device__ __host__ tuple<double,double,double,double,double,double>
////    operator()(Tuple t){
////        DisparityPoint p_disparity ;
////        p_disparity.u = get<0>(t) ;
////        p_disparity.v = get<1>(t) ;
////        p_disparity.d = get<2>(t) ;
////        p_disparity.vu = get<3>(t) ;
////        p_disparity.vv = get<4>(t) ;
////        p_disparity.vd = get<5>(t) ;
////        Extrinsics e = extrinsics[get<6>(t)] ;

////        EuclideanPoint p_world = model_.invertMeasurement(p_disparity,e) ;

//////        if (p_world.vx > max_v)
//////            p_world.vx = max_v ;
//////        if (p_world.vy > max_v)
//////            p_world.vy = max_v ;
//////        if (p_world.vz > max_v)
//////            p_world.vz = max_v ;

////        return make_tuple(p_world.x,p_world.y, p_world.z,
////                          p_world.vx,p_world.vy,p_world.vz) ;
////    }
////};

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

////struct aggregate_disparity_stats :
////    public thrust::binary_function<const DisparityStats&, const DisparityStats&,
////                                   DisparityStats>
////{
////    __host__ __device__
////    DisparityStats operator()(const DisparityStats &x, const DisparityStats &y){
////        DisparityStats result ;
////        int n = x.n + y.n ;
////        DisparityPoint delta = x.mean - y.mean ;
////        result.n = n ;
////        result.mean = x.mean + delta*y.n/n ;

////        result.variance = x.variance + y.variance ;
////        result.variance += delta*delta*x.n*y.n/n ;

////        result.cov_u_v = x.cov_u_v + y.cov_u_v ;
////        result.cov_u_v += delta.u*delta.v*x.n*y.n/n ;

////        result.cov_u_d = x.cov_u_d + y.cov_u_d ;
////        result.cov_u_d += delta.u*delta.d*x.n*y.n/n ;

////        result.cov_u_vu = x.cov_u_vu + y.cov_u_vu ;
////        result.cov_u_vu += delta.u*delta.vu*x.n*y.n/n ;

////        result.cov_u_vv = x.cov_u_vv + y.cov_u_vv ;
////        result.cov_u_vv += delta.u*delta.vv*x.n*y.n/n ;

////        result.cov_u_vd = x.cov_u_vd + y.cov_u_vd ;
////        result.cov_u_vd += delta.u*delta.vd*x.n*y.n/n ;

////        result.cov_v_d = x.cov_v_d + y.cov_v_d ;
////        result.cov_v_d += delta.v*delta.d*x.n*y.n/n ;

////        result.cov_v_vu = x.cov_v_vu + y.cov_v_vu ;
////        result.cov_v_vu += delta.v*delta.vu*x.n*y.n/n ;

////        result.cov_v_vv = x.cov_v_vv + y.cov_v_vv ;
////        result.cov_v_vv += delta.v*delta.vv*x.n*y.n/n ;

////        result.cov_v_vd = x.cov_v_vd + y.cov_v_vd ;
////        result.cov_v_vd += delta.v*delta.vd*x.n*y.n/n ;

////        result.cov_d_vu = x.cov_d_vu + y.cov_d_vu ;
////        result.cov_d_vu += delta.d*delta.vu*x.n*y.n/n ;

////        result.cov_d_vv = x.cov_d_vv + y.cov_d_vv ;
////        result.cov_d_vv += delta.d*delta.vv*x.n*y.n/n ;

////        result.cov_d_vd = x.cov_d_vd + y.cov_d_vd ;
////        result.cov_d_vd += delta.d*delta.vd*x.n*y.n/n ;

////        result.cov_vu_vv = x.cov_vu_vv + y.cov_vu_vv ;
////        result.cov_vu_vv += delta.vu*delta.vv*x.n*y.n/n ;

////        result.cov_vu_vd = x.cov_vu_vd + y.cov_vu_vd ;
////        result.cov_vu_vd += delta.vu*delta.vd*x.n*y.n/n ;

////        result.cov_vv_vd = x.cov_vv_vd + y.cov_vv_vd ;
////        result.cov_vv_vd += delta.vv*delta.vd*x.n*y.n/n ;

////        return result ;
////    }
////};

//struct update_components
//{
//    double* u_ ;
//    double* v_ ;
//    double* pd_ ;
//    Gaussian6D* features_ ;
//    DisparityMeasurementModel model_ ;
//    update_components(Gaussian6D* features, double* u, double* v, double* pd,
//                           DisparityMeasurementModel model) :
//        features_(features), u_(u), v_(v), model_(model), pd_(pd) {}

//    template <typename T>
//    __device__ void
//    operator()(T t){
//        // unpack tuple
//        int idx_feature = get<0>(t) ;
//        int idx_measure = get<1>(t) ;

//        Gaussian6D f = features_[idx_feature] ;
//        Gaussian6D f_update ;
//        double* p = f.cov ;
//        double pd = pd_[idx_feature] ;

//        double var_u = model_.std_u()*model_.std_u() ;
//        double var_v = model_.std_v()*model_.std_v() ;

//        // innovation vector
//        double innov[2] ;
//        innov[0] = u_[idx_measure] - f.mean[0] ;
//        innov[1] = v_[idx_measure] - f.mean[1] ;

//        // Innovation covariance
//        double sigma[4] ;
//        sigma[0] = p[0] + var_u;
//        sigma[1] = p[1];
//        sigma[2] = p[6];
//        sigma[3] = p[7] + var_v;

//        // enforce symmetry
//        sigma[1] = (sigma[1]+sigma[2])/2 ;
//        sigma[2] = sigma[1] ;

//        // inverse sigma
//        double s[4] ;
//        double det = sigma[0]*sigma[3] - sigma[1]*sigma[2] ;
//        s[0] = sigma[3]/det ;
//        s[1] = -sigma[1]/det ;
//        s[2] = -sigma[2]/det ;
//        s[3] = sigma[0]/det ;

//        // measurement likelihood
//        double dist = innov[0]*innov[0]*s[0] +
//                innov[0]*innov[1]*(s[1] + s[2]) +
//                innov[1]*innov[1]*s[3] ;
//        f_update.weight = safeLog(pd)
//                + safeLog(f.weight)
//                - 0.5*dist
//                - safeLog(2*M_PI)
//                - 0.5*safeLog(det) ;

//        // Kalman gain K = PH'/S
//        double K[12] ;
//        K[0] = p[0] * s[0] + p[6] * s[1];
//        K[1] = p[1] * s[0] + p[7] * s[1];
//        K[2] = p[2] * s[0] + p[8] * s[1];
//        K[3] = p[3] * s[0] + p[9] * s[1];
//        K[4] = p[4] * s[0] + p[10] * s[1];
//        K[5] = p[5] * s[0] + p[11] * s[1];
//        K[6] = p[0] * s[2] + p[6] * s[3];
//        K[7] = p[1] * s[2] + p[7] * s[3];
//        K[8] = p[2] * s[2] + p[8] * s[3];
//        K[9] = p[3] * s[2] + p[9] * s[3];
//        K[10] = p[4] * s[2] + p[10] * s[3];
//        K[11] = p[5] * s[2] + p[11] * s[3];

//        // updated mean x = x + K*innov
//        f_update.mean[0] = f.mean[0] + innov[0]*K[0] + innov[1]*K[6] ;
//        f_update.mean[1] = f.mean[1] + innov[0]*K[1] + innov[1]*K[7] ;
//        f_update.mean[2] = f.mean[2] + innov[0]*K[2] + innov[1]*K[8] ;
//        f_update.mean[3] = f.mean[3] + innov[0]*K[3] + innov[1]*K[9] ;
//        f_update.mean[4] = f.mean[4] + innov[0]*K[4] + innov[1]*K[10] ;
//        f_update.mean[5] = f.mean[5] + innov[0]*K[5] + innov[1]*K[11] ;

//        // updated covariance P = IKH*P/IKH' + KRK'

//        f_update.cov[0] = (1 - K[0]) * (p[0] * (1 - K[0]) - p[6] * K[6]) - K[6] * (p[1] * (1 - K[0]) - p[7] * K[6]) + var_u *  K[0]*K[0] + var_v * K[6]*K[6];
//        f_update.cov[1] = -K[1] * (p[0] * (1 - K[0]) - p[6] * K[6]) + (1 - K[7]) * (p[1] * (1 - K[0]) - p[7] * K[6]) + K[0] * var_u * K[1] + K[6] * var_v * K[7];
//        f_update.cov[2] = -K[2] * (p[0] * (1 - K[0]) - p[6] * K[6]) - K[8] * (p[1] * (1 - K[0]) - p[7] * K[6]) + p[2] * (1 - K[0]) - p[8] * K[6] + K[0] * var_u * K[2] + K[6] * var_v * K[8];
//        f_update.cov[3] = -K[3] * (p[0] * (1 - K[0]) - p[6] * K[6]) - K[9] * (p[1] * (1 - K[0]) - p[7] * K[6]) + p[3] * (1 - K[0]) - p[9] * K[6] + K[0] * var_u * K[3] + K[6] * var_v * K[9];
//        f_update.cov[4] = -K[4] * (p[0] * (1 - K[0]) - p[6] * K[6]) - K[10] * (p[1] * (1 - K[0]) - p[7] * K[6]) + p[4] * (1 - K[0]) - p[10] * K[6] + K[0] * var_u * K[4] + K[6] * var_v * K[10];
//        f_update.cov[5] = -K[5] * (p[0] * (1 - K[0]) - p[6] * K[6]) - K[11] * (p[1] * (1 - K[0]) - p[7] * K[6]) + p[5] * (1 - K[0]) - p[11] * K[6] + K[0] * var_u * K[5] + K[6] * var_v * K[11];
//        f_update.cov[6] = (1 - K[0]) * (-p[0] * K[1] + p[6] * (1 - K[7])) - K[6] * (-p[1] * K[1] + p[7] * (1 - K[7])) + K[0] * var_u * K[1] + K[6] * var_v * K[7];
//        f_update.cov[7] = -K[1] * (-p[0] * K[1] + p[6] * (1 - K[7])) + (1 - K[7]) * (-p[1] * K[1] + p[7] * (1 - K[7])) + var_u *  K[1]*K[1] + var_v *  K[7]*K[7];
//        f_update.cov[8] = -K[2] * (-p[0] * K[1] + p[6] * (1 - K[7])) - K[8] * (-p[1] * K[1] + p[7] * (1 - K[7])) - p[2] * K[1] + p[8] * (1 - K[7]) + K[1] * var_u * K[2] + K[7] * var_v * K[8];
//        f_update.cov[9] = -K[3] * (-p[0] * K[1] + p[6] * (1 - K[7])) - K[9] * (-p[1] * K[1] + p[7] * (1 - K[7])) - p[3] * K[1] + p[9] * (1 - K[7]) + K[1] * var_u * K[3] + K[7] * var_v * K[9];
//        f_update.cov[10] = -K[4] * (-p[0] * K[1] + p[6] * (1 - K[7])) - K[10] * (-p[1] * K[1] + p[7] * (1 - K[7])) - p[4] * K[1] + p[10] * (1 - K[7]) + K[1] * var_u * K[4] + K[7] * var_v * K[10];
//        f_update.cov[11] = -K[5] * (-p[0] * K[1] + p[6] * (1 - K[7])) - K[11] * (-p[1] * K[1] + p[7] * (1 - K[7])) - p[5] * K[1] + p[11] * (1 - K[7]) + K[1] * var_u * K[5] + K[7] * var_v * K[11];
//        f_update.cov[12] = (1 - K[0]) * (-p[0] * K[2] - p[6] * K[8] + p[12]) - K[6] * (-p[1] * K[2] - p[7] * K[8] + p[13]) + K[0] * var_u * K[2] + K[6] * var_v * K[8];
//        f_update.cov[13] = -K[1] * (-p[0] * K[2] - p[6] * K[8] + p[12]) + (1 - K[7]) * (-p[1] * K[2] - p[7] * K[8] + p[13]) + K[1] * var_u * K[2] + K[7] * var_v * K[8];
//        f_update.cov[14] = -K[2] * (-p[0] * K[2] - p[6] * K[8] + p[12]) - K[8] * (-p[1] * K[2] - p[7] * K[8] + p[13]) - p[2] * K[2] - p[8] * K[8] + p[14] + var_u * K[2]*K[2] + var_v * K[8]*K[8];
//        f_update.cov[15] = -K[3] * (-p[0] * K[2] - p[6] * K[8] + p[12]) - K[9] * (-p[1] * K[2] - p[7] * K[8] + p[13]) - p[3] * K[2] - p[9] * K[8] + p[15] + K[2] * var_u * K[3] + K[8] * var_v * K[9];
//        f_update.cov[16] = -K[4] * (-p[0] * K[2] - p[6] * K[8] + p[12]) - K[10] * (-p[1] * K[2] - p[7] * K[8] + p[13]) - p[4] * K[2] - p[10] * K[8] + p[16] + K[2] * var_u * K[4] + K[8] * var_v * K[10];
//        f_update.cov[17] = -K[5] * (-p[0] * K[2] - p[6] * K[8] + p[12]) - K[11] * (-p[1] * K[2] - p[7] * K[8] + p[13]) - p[5] * K[2] - p[11] * K[8] + p[17] + K[2] * var_u * K[5] + K[8] * var_v * K[11];
//        f_update.cov[18] = (1 - K[0]) * (-p[0] * K[3] - p[6] * K[9] + p[18]) - K[6] * (-p[1] * K[3] - p[7] * K[9] + p[19]) + K[0] * var_u * K[3] + K[6] * var_v * K[9];
//        f_update.cov[19] = -K[1] * (-p[0] * K[3] - p[6] * K[9] + p[18]) + (1 - K[7]) * (-p[1] * K[3] - p[7] * K[9] + p[19]) + K[1] * var_u * K[3] + K[7] * var_v * K[9];
//        f_update.cov[20] = -K[2] * (-p[0] * K[3] - p[6] * K[9] + p[18]) - K[8] * (-p[1] * K[3] - p[7] * K[9] + p[19]) - p[2] * K[3] - p[8] * K[9] + p[20] + K[2] * var_u * K[3] + K[8] * var_v * K[9];
//        f_update.cov[21] = -K[3] * (-p[0] * K[3] - p[6] * K[9] + p[18]) - K[9] * (-p[1] * K[3] - p[7] * K[9] + p[19]) - p[3] * K[3] - p[9] * K[9] + p[21] + var_u *  K[3]*K[3] + var_v * K[9]*K[9];
//        f_update.cov[22] = -K[4] * (-p[0] * K[3] - p[6] * K[9] + p[18]) - K[10] * (-p[1] * K[3] - p[7] * K[9] + p[19]) - p[4] * K[3] - p[10] * K[9] + p[22] + K[3] * var_u * K[4] + K[9] * var_v * K[10];
//        f_update.cov[23] = -K[5] * (-p[0] * K[3] - p[6] * K[9] + p[18]) - K[11] * (-p[1] * K[3] - p[7] * K[9] + p[19]) - p[5] * K[3] - p[11] * K[9] + p[23] + K[3] * var_u * K[5] + K[9] * var_v * K[11];
//        f_update.cov[24] = (1 - K[0]) * (-p[0] * K[4] - p[6] * K[10] + p[24]) - K[6] * (-p[1] * K[4] - p[7] * K[10] + p[25]) + K[0] * var_u * K[4] + K[6] * var_v * K[10];
//        f_update.cov[25] = -K[1] * (-p[0] * K[4] - p[6] * K[10] + p[24]) + (1 - K[7]) * (-p[1] * K[4] - p[7] * K[10] + p[25]) + K[1] * var_u * K[4] + K[7] * var_v * K[10];
//        f_update.cov[26] = -K[2] * (-p[0] * K[4] - p[6] * K[10] + p[24]) - K[8] * (-p[1] * K[4] - p[7] * K[10] + p[25]) - p[2] * K[4] - p[8] * K[10] + p[26] + K[2] * var_u * K[4] + K[8] * var_v * K[10];
//        f_update.cov[27] = -K[3] * (-p[0] * K[4] - p[6] * K[10] + p[24]) - K[9] * (-p[1] * K[4] - p[7] * K[10] + p[25]) - p[3] * K[4] - p[9] * K[10] + p[27] + K[3] * var_u * K[4] + K[9] * var_v * K[10];
//        f_update.cov[28] = -K[4] * (-p[0] * K[4] - p[6] * K[10] + p[24]) - K[10] * (-p[1] * K[4] - p[7] * K[10] + p[25]) - p[4] * K[4] - p[10] * K[10] + p[28] + var_u * K[4]*K[4] + var_v *  K[10]*K[10];
//        f_update.cov[29] = -K[5] * (-p[0] * K[4] - p[6] * K[10] + p[24]) - K[11] * (-p[1] * K[4] - p[7] * K[10] + p[25]) - p[5] * K[4] - p[11] * K[10] + p[29] + K[4] * var_u * K[5] + K[10] * var_v * K[11];
//        f_update.cov[30] = (1 - K[0]) * (-p[0] * K[5] - p[6] * K[11] + p[30]) - K[6] * (-p[1] * K[5] - p[7] * K[11] + p[31]) + K[0] * var_u * K[5] + K[6] * var_v * K[11];
//        f_update.cov[31] = -K[1] * (-p[0] * K[5] - p[6] * K[11] + p[30]) + (1 - K[7]) * (-p[1] * K[5] - p[7] * K[11] + p[31]) + K[1] * var_u * K[5] + K[7] * var_v * K[11];
//        f_update.cov[32] = -K[2] * (-p[0] * K[5] - p[6] * K[11] + p[30]) - K[8] * (-p[1] * K[5] - p[7] * K[11] + p[31]) - p[2] * K[5] - p[8] * K[11] + p[32] + K[2] * var_u * K[5] + K[8] * var_v * K[11];
//        f_update.cov[33] = -K[3] * (-p[0] * K[5] - p[6] * K[11] + p[30]) - K[9] * (-p[1] * K[5] - p[7] * K[11] + p[31]) - p[3] * K[5] - p[9] * K[11] + p[33] + K[3] * var_u * K[5] + K[9] * var_v * K[11];
//        f_update.cov[34] = -K[4] * (-p[0] * K[5] - p[6] * K[11] + p[30]) - K[10] * (-p[1] * K[5] - p[7] * K[11] + p[31]) - p[4] * K[5] - p[10] * K[11] + p[34] + K[4] * var_u * K[5] + K[10] * var_v * K[11];
//        f_update.cov[35] = -K[5] * (-p[0] * K[5] - p[6] * K[11] + p[30]) - K[11] * (-p[1] * K[5] - p[7] * K[11] + p[31]) - p[5] * K[5] - p[11] * K[11] + p[35] + var_u * K[5]*K[5] + var_v *  K[11]*K[11];

//        get<2>(t) = f_update ;
//    }
//};

////struct compute_birth{
////    DisparityPoint birth_means ;
////    DisparityPoint birth_vars ;
////    double w0 ;

////    compute_birth(DisparityPoint means0, DisparityPoint vars0, double w0) :
////        birth_means(means0), birth_vars(vars0), w0(w0)
////    {}

////    template<typename T>
////    __device__ void
////    operator()(T t){
////        double u = get<0>(t) ;
////        double v = get<1>(t) ;
////        Gaussian6D feature_birth ;
////        feature_birth.mean[0] = u ;
////        feature_birth.mean[1] = v ;
////        feature_birth.mean[2] = birth_means.d ;
////        feature_birth.mean[3] = birth_means.vu ;
////        feature_birth.mean[4] = birth_means.vv ;
////        feature_birth.mean[5] = birth_means.vd ;

////        for ( int i = 0 ; i < 36 ; i++)
////            feature_birth.cov[i] = 0 ;
////        feature_birth.cov[0] = birth_vars.u ;
////        feature_birth.cov[7] = birth_vars.v ;
////        feature_birth.cov[14] = birth_vars.d ;
////        feature_birth.cov[21] = birth_vars.vu ;
////        feature_birth.cov[28] = birth_vars.vv ;
////        feature_birth.cov[35] = birth_vars.vd ;

////        feature_birth.weight = safeLog(w0) ;

////        get<2>(t) = feature_birth ;
////    }
////};

//struct compute_nondetect{

//    template <typename T>
//    __device__ void
//    operator()(T t){
//        Gaussian6D feature_predict = get<0>(t) ;
//        double pd = get<1>(t) ;

//        Gaussian6D feature_nondetect = feature_predict ;
//        feature_nondetect.weight = feature_predict.weight*(1-pd) ;
//        get<2>(t) = feature_nondetect ;
//    }
//};


//struct log_sum_exp : public thrust::binary_function<const double, const double, double>
//{
//    __device__ __host__ double
//    operator()(const double a, const double b){
//        if (a > b){
//            return a + safeLog(1 + exp(b-a)) ;
//        }
//        else{
//            return b + safeLog(1 + exp(a-b)) ;
//        }
//    }
//} ;

///// extract the weight
//struct get_weight : public thrust::unary_function<const Gaussian6D, double>
//{
//    __device__ __host__ double
//    operator()(const Gaussian6D g){
//        return g.weight ;
//    }
//} ;

///// subtract from a constant
//template <typename T>
//struct subtract_from : public thrust::unary_function<T,T>
//{
//    T val_ ;
//    __device__ __host__ subtract_from(T val) : val_(val) {}

//    __device__ __host__ T
//    operator()(T x){ return ( val_ - x) ; }
//} ;

///// multiply by 1-pd
//struct nondetect_weight : public thrust::binary_function<double,double,double>
//{
//    __device__ __host__ double
//    operator()(double w, double pd){
//        return w*(1-pd) ;
//    }
//} ;


///// multiply by a constant
//struct times : public thrust::unary_function<const int, int>
//{
//    const int C_ ;
//    times(int c) : C_(c) {}

//    __device__ __host__ int
//    operator()(const int x){ return C_*x ; }
//} ;

//template <typename T1, typename T2>
//struct divide_by : public thrust::unary_function<T1,T2>
//{
//    const T2 C_ ;
//    divide_by(T2 c) : C_(c) {}

//    __device__ __host__ T2
//    operator()(const T1 x){ return x/C_ ; }
//} ;

//template <typename T1, typename T2>
//struct exponentiate : public thrust::unary_function<T1,T2>
//{
//    __device__ __host__ T2
//    operator()(T1 x){ return exp(x) ; }
//} ;

//template <typename T>
//struct square : public thrust::unary_function<T,T>
//{
//    __device__ __host__ T
//    operator()(T x) { return x*x ; }
//} ;

//template <typename T>
//struct equals : public thrust::unary_function<T,bool>
//{
//    T val ;
//    __device__ __host__ equals(T val) : val(val) {}

//    __device__ __host__ bool
//    operator()(T x) { return (x == val) ; }
//} ;

//template <typename T>
//struct gt : public thrust::unary_function<T,bool>
//{
//    T val ;
//    __device__ __host__ gt(T val) : val(val) {}

//    __device__ __host__ bool
//    operator()(T x) { return ( x > val ) ; }
//} ;

//template <typename T>
//struct lt : public thrust::unary_function<T,bool>
//{
//    T val ;
//    __device__ __host__ lt(T val) : val(val) {}

//    __device__ __host__ bool
//    operator()(T x) { return ( x < val ) ; }
//} ;

//template <typename T>
//struct geq : public thrust::unary_function<T,bool>
//{
//    T val ;
//    __device__ __host__ geq(T val) : val(val) {}

//    __device__ __host__ bool
//    operator()(T x) { return ( x >= val ) ; }
//} ;

//template <typename T>
//struct leq : public thrust::unary_function<T,bool>
//{
//    T val ;
//    __device__ __host__ leq(T val) : val(val) {}

//    __device__ __host__ bool
//    operator()(T x) { return ( x <= val ) ; }
//} ;

//struct sample_disparity_gaussian
//{
//    const int n_samples_ ;
//    const Gaussian6D* gaussians_ ;
//    double seed ;
//    sample_disparity_gaussian(Gaussian6D* g, int n, int s) : n_samples_(n), gaussians_(g), seed(s) {}

//    // number of parallel threads is n_features*samples_per_feature
//    // tuple argument is (tid, u, v, d, vu, vv, vd) ;
//    template<typename T>
//    __device__ void
//    operator()(T t){
//        int tid = get<0>(t) ;

//        // generate uncorrelated normally distributed values
//        thrust::default_random_engine rng(seed) ;
//        rng.discard(6*tid) ;
//        thrust::random::normal_distribution<double> randn(0.0,1.0) ;
//        double vals[6] ;
//        for ( int i = 0 ; i < 6 ; i++)
//            vals[i] = randn(rng) ;

////        printf("%d: %f %f %f %f %f %f\n", tid, vals[0],vals[1],vals[2],
////                vals[3],vals[4],vals[5]) ;
//        // uncorrelated values are transformed by multiplying with cholesky
//        // decomposition of covariance matrix, and adding mean
//        int idx = int(tid/n_samples_) ;
//        Gaussian6D feature = gaussians_[idx] ;

//        double L[36] ;
//        cholesky(feature.cov,L,6);

//        get<1>(t) = L[0]*vals[0] + feature.mean[0] ;
//        get<2>(t) = L[1]*vals[0] + L[7]*vals[1] + feature.mean[1] ;
//        get<3>(t) = L[2]*vals[0] + L[8]*vals[1] + L[14]*vals[2]
//                + feature.mean[2] ;
//        get<4>(t) = L[3]*vals[0] + L[9]*vals[1] + L[15]*vals[2]
//                + L[21]*vals[3] + feature.mean[3] ;
//        get<5>(t) = L[4]*vals[0] + L[10]*vals[1] + L[16]*vals[2]
//                + L[22]*vals[3] + L[28]*vals[4] + feature.mean[4] ;
//        get<6>(t) = L[5]*vals[0] + L[11]*vals[1] + L[17]*vals[2]
//                + L[23]*vals[3] + L[29]*vals[4] + L[35]*vals[5]
//                + feature.mean[5] ;
//    }
//};
