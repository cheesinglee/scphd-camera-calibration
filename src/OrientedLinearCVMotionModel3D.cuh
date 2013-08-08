#ifndef ORIENTEDLINEARCVMOTIONMODEL3D_H
#define ORIENTEDLINEARCVMOTIONMODEL3D_H

#include "linearcvmotionmodel3d.cuh"
#include "types.h"

// this class is essentially a wrapper for a pair of constant velocity motion
// models - one for the cartesian portion of the state and another for the
// angular portion

class OrientedLinearCVMotionModel3D
{
public:
    OrientedLinearCVMotionModel3D(){} ;

    OrientedLinearCVMotionModel3D(double ax, double ay, double az,
                                  double ax_angular, double ay_angular,
                                  double az_angular)
    {
        cartesian_model_ = LinearCVMotionModel3D(ax,ay,az) ;
        angular_model_ = LinearCVMotionModel3D(ax_angular,
                                               ay_angular,
                                               az_angular) ;
    }

    __host__ __device__
    void computeMotion(EuclideanPoint& cartesian_state,
                       EuclideanPoint& angular_state,
                       double dt){
        cartesian_state = cartesian_model_.computeMotion(cartesian_state,dt) ;
        angular_state = angular_model_.computeMotion(angular_state,dt) ;
    }

    __host__ __device__
    void computeNoisyMotion(EuclideanPoint& cartesian_state,
                       EuclideanPoint& angular_state,
                       double dt,
                       double ax_c, double ay_c, double az_c,
                       double ax_a, double ay_a, double az_a){
        cartesian_state = cartesian_model_.computeNoisyMotion(
                    cartesian_state,dt,ax_c,ay_c,az_c) ;
        angular_state = angular_model_.computeNoisyMotion(
                    angular_state,dt,ax_a,ay_a,az_a) ;
    }

    double std_ax_cartesian() { return cartesian_model_.std_ax() ; }
    double std_ay_cartesian() { return cartesian_model_.std_ay() ; }
    double std_az_cartesian() { return cartesian_model_.std_az() ; }

    double std_ax_angular() { return angular_model_.std_ax() ; }
    double std_ay_angular() { return angular_model_.std_ay() ; }
    double std_az_angular() { return angular_model_.std_az() ; }

private:
    LinearCVMotionModel3D cartesian_model_ ;
    LinearCVMotionModel3D angular_model_ ;
};

#endif // ORIENTEDLINEARCVMOTIONMODEL3D_H
