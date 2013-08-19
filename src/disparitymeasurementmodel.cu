#include "disparitymeasurementmodel.cuh"
#include "stdio.h"
__host__ __device__
EuclideanPoint
transformWorldToCamera(EuclideanPoint p_world, Extrinsics e){
    double X = p_world.x ;
    double Y = p_world.y ;
    double Z = p_world.z ;
    double VX = p_world.vx ;
    double VY = p_world.vy ;
    double VZ = p_world.vz ;

    double x = e.cartesian.x ;
    double y = e.cartesian.y;
    double z = e.cartesian.z ;
    double ctheta = cos(e.angular.x) ;
    double stheta = sin(e.angular.x) ;
    double cphi = cos(e.angular.y) ;
    double sphi = sin(e.angular.y) ;
    double cpsi = cos(e.angular.z) ;
    double spsi = sin(e.angular.z) ;

    EuclideanPoint p_camera ;

    p_camera.x = cphi * cpsi * X
            + (ctheta * spsi + stheta * sphi * cpsi) * Y
            + (stheta * spsi - ctheta * sphi * cpsi) * Z
            - cphi*cpsi*x
            - (ctheta*spsi + stheta*sphi*cpsi)*y
            - (stheta*spsi - ctheta*sphi*cpsi)*z;
    p_camera.y = -cphi * spsi * X
            + (ctheta * cpsi - stheta * sphi * spsi) * Y
            + (stheta * cpsi + ctheta * sphi * spsi) * Z
            - (-cphi*spsi)*x
            - (ctheta * cpsi - stheta * sphi * spsi) * y
            - (stheta * cpsi + ctheta * sphi * spsi) * z ;
    p_camera.z = sphi * X
            - stheta * cphi * Y
            + ctheta * cphi * Z
            - sphi *x
            - (-stheta*cphi)*y
            - (ctheta*cphi)*z ;

    p_camera.vx = cphi * cpsi * VX
            + (ctheta * spsi + stheta * sphi * cpsi) * VY
            + (stheta * spsi - ctheta * sphi * cpsi) * VZ;
    p_camera.vy = -cphi * spsi * VX
            + (ctheta * cpsi - stheta * sphi * spsi) * VY
            + (stheta * cpsi + ctheta * sphi * spsi) * VZ ;
    p_camera.vz = sphi * VX
            - stheta * cphi * VY
            + ctheta * cphi * VZ ;


//    p_camera.x = cphi * cpsi * X - cphi * spsi * Y +
//            sphi * Z - cphi * cpsi * x +
//            cphi * spsi * y - sphi * z;
//    p_camera.y = (ctheta * spsi + stheta * sphi * cpsi) * X
//            + (ctheta * cpsi - stheta * sphi * spsi) * Y
//            - stheta * cphi * Z
//            - (ctheta * spsi + stheta * sphi * cpsi) * x
//            - (ctheta * cpsi - stheta * sphi * spsi) * y
//            + stheta * cphi * z;
//    p_camera.z = (stheta * spsi - ctheta * sphi * cpsi) * X
//            + (stheta * cpsi + ctheta * sphi * spsi) * Y
//            + ctheta * cphi * Z
//            - (stheta * spsi - ctheta * sphi * cpsi) * x
//            - (stheta * cpsi + ctheta * sphi * spsi) * y
//            - ctheta * cphi * z;
//    p_camera.vx = cphi * cpsi * VX - cphi * spsi * VY
//            + sphi * VZ ;
//    p_camera.vy = (ctheta * spsi + stheta * sphi * cpsi) * VX
//            + (ctheta * cpsi - stheta * sphi * spsi) * VY
//            - stheta * cphi * VZ ;
//    p_camera.vz = (stheta * spsi - ctheta * sphi * cpsi) * VX
//            + (stheta * cpsi + ctheta * sphi * spsi) * VY
//            + ctheta * cphi * VZ ;

    return p_camera ;
}

__host__ __device__
EuclideanPoint
transformCameraToWorld(EuclideanPoint p_camera, Extrinsics e){
    double X = p_camera.x ;
    double Y = p_camera.y ;
    double Z = p_camera.z ;
    double VX = p_camera.vx ;
    double VY = p_camera.vy ;
    double VZ = p_camera.vz ;

    EuclideanPoint p_world ;
    double x = e.cartesian.x ;
    double y = e.cartesian.y;
    double z = e.cartesian.z ;
    double ctheta = cos(e.angular.x) ;
    double stheta = sin(e.angular.x) ;
    double cphi = cos(e.angular.y) ;
    double sphi = sin(e.angular.y) ;
    double cpsi = cos(e.angular.z) ;
    double spsi = sin(e.angular.z) ;

    p_world.x = cphi * cpsi * X - cphi * spsi * Y +
            sphi * Z + x;
    p_world.y = (ctheta * spsi + stheta * sphi * cpsi) * X
            + (ctheta * cpsi - stheta * sphi * spsi) * Y
            - stheta * cphi * Z + y;
    p_world.z = (stheta * spsi - ctheta * sphi * cpsi) * X
            + (stheta * cpsi + ctheta * sphi * spsi) * Y
            + ctheta * cphi * Z + z;

    p_world.vx = cphi * cpsi * VX - cphi * spsi * VY
            + sphi * VZ ;
    p_world.vy = (ctheta * spsi + stheta * sphi * cpsi) * VX
            + (ctheta * cpsi - stheta * sphi * spsi) * VY
            - stheta * cphi * VZ ;
    p_world.vz = (stheta * spsi - ctheta * sphi * cpsi) * VX
            + (stheta * cpsi + ctheta * sphi * spsi) * VY
            + ctheta * cphi * VZ ;

//    p_world.x = cphi * cpsi * X
//            + (ctheta * spsi + stheta * sphi * cpsi) * Y
//            + (stheta * spsi - ctheta * sphi * cpsi) * Z
//            + x;
//    p_world.y = -cphi * spsi * X
//            + (ctheta * cpsi - stheta * sphi * spsi) * Y
//            + (stheta * cpsi + ctheta * sphi * spsi) * Z
//            + y;
//    p_world.z = sphi * X
//            - stheta * cphi * Y
//            + ctheta * cphi * Z
//            + z;

//    p_world.vx = cphi * cpsi * VX
//            + (ctheta * spsi + stheta * sphi * cpsi) * VY
//            + (stheta * spsi - ctheta * sphi * cpsi) * VZ ;
//    p_world.vy = -cphi * spsi * VX
//            + (ctheta * cpsi - stheta * sphi * spsi) * VY
//            + (stheta * cpsi + ctheta * sphi * spsi) * VZ ;
//    p_world.vz = sphi * VX
//            - stheta * cphi * VY
//            + ctheta * cphi * VZ ;
    return p_world ;
}


__host__ __device__
DisparityMeasurementModel::DisparityMeasurementModel()
    : fx_(1000), fy_(1000), u0_(250), v0_(250),
      std_u_(1), std_v_(1),
      img_height_(500),img_width_(500), pd_(0.99)
{}


__host__ __device__
DisparityMeasurementModel::DisparityMeasurementModel(double fx, double fy, double u0, double v0, double std_u, double std_v, double pd, double lambda)
    : fx_(fx), fy_(fy), u0_(u0), v0_(v0), std_u_(std_u), std_v_(std_v), pd_(pd)
{
    img_height_ = (2*v0) ;
    img_width_= (2*u0);
    kappa_ = lambda / (img_height_*img_height_) ;
}


__host__ __device__ DisparityPoint
DisparityMeasurementModel::computeMeasurement(EuclideanPoint p_world){
    double x = p_world.x ;
    double y = p_world.y ;
    double z = p_world.z ;
    double vx = p_world.vx ;
    double vy = p_world.vy ;
    double vz = p_world.vz ;

    DisparityPoint p ;
    p.u = u0_ - fx_/z*x ;
    p.v = v0_ - fy_/z*y ;
    p.d = -fx_ / z ;
    p.vu = -fx_*(vx*z - vz*x)/(z*z) ;
    p.vv = -fy_*(vy*z - vz*y)/(z*z) ;
    p.vd = fx_*vz/(z*z) ;
    return p ;
}


__host__ __device__ DisparityPoint
DisparityMeasurementModel::computeMeasurement(EuclideanPoint p_world, Extrinsics e){
    EuclideanPoint p_camera = transformWorldToCamera(p_world,e) ;
//    printf ("world_point = %f %f %f\ncamera point = %f %f %f\n",
//            p_world.x,
//            p_world.y,
//            p_world.z,
//            p_camera.x,
//            p_camera.y,
//            p_camera.z) ;
    return computeMeasurement(p_camera) ;
}

__host__ __device__ EuclideanPoint
DisparityMeasurementModel::invertMeasurement(DisparityPoint p_disparity, Extrinsics e){
    double u = p_disparity.u ;
    double v = p_disparity.v ;
    double d = p_disparity.d ;
    double vu = p_disparity.vu ;
    double vv = p_disparity.vv ;
    double vd = p_disparity.vd ;

    EuclideanPoint p_camera ;
    p_camera.x = (u - u0_)/d ;
    p_camera.y = (fx_/fy_)*(v - v0_)/d ;
    p_camera.z = -fx_/d ;
    p_camera.vx = (vu*d - vd*(u-u0_))/(d*d) ;
    p_camera.vy = (fx_/fy_)*(vv*d - vd*(v-v0_))/(d*d) ;
    p_camera.vz = fx_*vd/(d*d) ;

    EuclideanPoint p_world = transformCameraToWorld(p_camera,e) ;
    return p_world ;
}

//__host__ __device__ bool
//DisparityMeasurementModel::checkVisibility(EuclideanPoint p_world, Extrinsics e){

//}
