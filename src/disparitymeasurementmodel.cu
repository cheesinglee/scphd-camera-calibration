#include "disparitymeasurementmodel.cuh"

__host__ __device__
EuclideanPoint
transformWorldToCamera(EuclideanPoint p_world, Extrinsics e){
    float X = p_world.x ;
    float Y = p_world.y ;
    float Z = p_world.z ;
    float VX = p_world.vx ;
    float VY = p_world.vy ;
    float VZ = p_world.vz ;

    float x = e.cartesian.x ;
    float y = e.cartesian.y;
    float z = e.cartesian.z ;
    float ctheta = cos(e.angular.x) ;
    float stheta = sin(e.angular.x) ;
    float cphi = cos(e.angular.y) ;
    float sphi = sin(e.angular.y) ;
    float cpsi = cos(e.angular.z) ;
    float spsi = sin(e.angular.z) ;

    EuclideanPoint p_camera ;
    p_camera.x = cphi * cpsi * X - cphi * spsi * Y +
            sphi * Z - cphi * cpsi * x +
            cphi * spsi * y - sphi * z;
    p_camera.y = (ctheta * spsi + stheta * sphi * cpsi) * X
            + (ctheta * cpsi - stheta * sphi * spsi) * Y
            - stheta * cphi * Z
            - (ctheta * spsi + stheta * sphi * cpsi) * x
            - (ctheta * cpsi - stheta * sphi * spsi) * y
            + stheta * cphi * z;
    p_camera.z = (stheta * spsi - ctheta * sphi * cpsi) * X
            + (stheta * cpsi + ctheta * sphi * spsi) * Y
            + ctheta * cphi * Z
            - (stheta * spsi - ctheta * sphi * cpsi) * x
            - (stheta * cpsi + ctheta * sphi * spsi) * y
            - ctheta * cphi * z;
    p_camera.vx = cphi * cpsi * VX - cphi * spsi * VY
            + sphi * VZ ;
    p_camera.vy = (ctheta * spsi + stheta * sphi * cpsi) * VX
            + (ctheta * cpsi - stheta * sphi * spsi) * VY
            - stheta * cphi * VZ ;
    p_camera.z = (stheta * spsi - ctheta * sphi * cpsi) * VX
            + (stheta * cpsi + ctheta * sphi * spsi) * VY
            + ctheta * cphi * VZ ;

    return p_camera ;
}

__host__ __device__
EuclideanPoint
transformCameraToWorld(EuclideanPoint p_camera, Extrinsics e){
    float X = p_camera.x ;
    float Y = p_camera.y ;
    float Z = p_camera.z ;
    float VX = p_camera.vx ;
    float VY = p_camera.vy ;
    float VZ = p_camera.vz ;

    EuclideanPoint p_world ;
    float x = e.cartesian.x ;
    float y = e.cartesian.y;
    float z = e.cartesian.z ;
    float ctheta = cos(e.angular.x) ;
    float stheta = sin(e.angular.x) ;
    float cphi = cos(e.angular.y) ;
    float sphi = sin(e.angular.y) ;
    float cpsi = cos(e.angular.z) ;
    float spsi = sin(e.angular.z) ;

    p_world.x = cphi * cpsi * X
            + (ctheta * spsi + stheta * sphi * cpsi) * Y
            + (stheta * spsi - ctheta * sphi * cpsi) * Z
            + x;
    p_world.y = -cphi * spsi * X
            + (ctheta * cpsi - stheta * sphi * spsi) * Y
            + (stheta * cpsi + ctheta * sphi * spsi) * Z
            + y;
    p_world.z = sphi * X
            - stheta * cphi * Y
            + ctheta * cphi * Z
            + z;

    p_world.vx = cphi * cpsi * VX
            + (ctheta * spsi + stheta * sphi * cpsi) * VY
            + (stheta * spsi - ctheta * sphi * cpsi) * VZ ;
    p_world.vy = -cphi * spsi * VX
            + (ctheta * cpsi - stheta * sphi * spsi) * VY
            + (stheta * cpsi + ctheta * sphi * spsi) * VZ ;
    p_world.vz = sphi * VX
            - stheta * cphi * VY
            + ctheta * cphi * VZ ;
    return p_world ;
}


__host__ __device__
DisparityMeasurementModel::DisparityMeasurementModel()
    : fx_(1000), fy_(1000), u0_(250), v0_(250),
      std_u_(1), std_v_(1),
      img_height_(500),img_width_(500), pd_(0.99)
{}


__host__ __device__
DisparityMeasurementModel::DisparityMeasurementModel(float fx, float fy, float u0, float v0, float std_u, float std_v, float pd, float lambda)
    : fx_(fx), fy_(fy), u0_(u0), v0_(v0), std_u_(std_u), std_v_(std_v), pd_(pd)
{
    img_height_ = (2*v0) ;
    img_width_= (2*u0);
    kappa_ = lambda / (img_height_*img_height_) ;
}


__host__ __device__ DisparityPoint
DisparityMeasurementModel::computeMeasurement(EuclideanPoint p_world){
    float x = p_world.x ;
    float y = p_world.y ;
    float z = p_world.z ;
    float vx = p_world.vx ;
    float vy = p_world.vy ;
    float vz = p_world.vz ;

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
    return computeMeasurement(p_camera) ;
}

__host__ __device__ EuclideanPoint
DisparityMeasurementModel::invertMeasurement(DisparityPoint p_disparity, Extrinsics e){
    float u = p_disparity.u ;
    float v = p_disparity.v ;
    float d = p_disparity.d ;
    float vu = p_disparity.vu ;
    float vv = p_disparity.vv ;
    float vd = p_disparity.vd ;

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
