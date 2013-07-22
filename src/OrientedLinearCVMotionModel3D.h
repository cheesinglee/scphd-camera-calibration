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
    OrientedLinearCVMotionModel3D(float ax, float ay, float az,
                                  float ax_angular, float ay_angular,
                                  float az_angular)
    {
        cartesian_model_ = LinearCVMotionModel3D(ax,ay,az) ;
        angular_model_ = LinearCVMotionModel3D(ax_angular,
                                               ay_angular,
                                               az_angular) ;
    }

    void computeMotion(EuclideanPoint& cartesian_state,
                       EuclideanPoint& angular_state,
                       float dt){
        cartesian_state = cartesian_model_.computeMotion(cartesian_state,dt) ;
        angular_state = angular_model_.computeMotion(angular_state,dt) ;
    }

    void computeNoisyMotion(EuclideanPoint& cartesian_state,
                       EuclideanPoint& angular_state,
                       float dt,
                       float ax_c, float ay_c, float az_c,
                       float ax_a, float ay_a, float az_a){
        cartesian_state = cartesian_model_.computeNoisyMotion(
                    cartesian_state,dt,ax_c,ay_c,az_c) ;
        angular_state = angular_model_.computeNoisyMotion(
                    angular_state,dt,ax_a,ay_a,az_a) ;
    }

    float std_ax_cartesian() { return cartesian_model_.std_ax() ; }
    float std_ay_cartesian() { return cartesian_model_.std_ay() ; }
    float std_az_cartesian() { return cartesian_model_.std_az() ; }

    float std_ax_angular() { return angular_model_.std_ax() ; }
    float std_ay_angular() { return angular_model_.std_ay() ; }
    float std_az_angular() { return angular_model_.std_az() ; }

private:
    LinearCVMotionModel3D cartesian_model_ ;
    LinearCVMotionModel3D angular_model_ ;
};

#endif // ORIENTEDLINEARCVMOTIONMODEL3D_H
