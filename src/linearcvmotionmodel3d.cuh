#ifndef LINEARCVMOTIONMODEL3D_CUH
#define LINEARCVMOTIONMODEL3D_CUH

#include "types.h"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

class LinearCVMotionModel3D{
public:
    LinearCVMotionModel3D() : std_ax_(1.0), std_ay_(1.0), std_az_(1.0) {}

    LinearCVMotionModel3D(double ax, double ay, double az) :
        std_ax_(ax), std_ay_(ay), std_az_(az){}

    CUDA_CALLABLE_MEMBER
    EuclideanPoint computeMotion(EuclideanPoint old_state, double dt){
        EuclideanPoint new_state = old_state ;
        new_state.x = old_state.x + old_state.vx*dt ;
        new_state.y = old_state.y + old_state.vy*dt ;
        new_state.z = old_state.z + old_state.vz*dt ;
        return new_state ;
    }

    CUDA_CALLABLE_MEMBER
    EuclideanPoint computeNoisyMotion(EuclideanPoint old_state, double dt,
                                       double ax, double ay, double az){
        EuclideanPoint clean_state = computeMotion(old_state,dt) ;
        EuclideanPoint noisy_state = clean_state ;
        noisy_state.x += 0.5*ax*dt*dt ;
        noisy_state.y += 0.5*ay*dt*dt ;
        noisy_state.z += 0.5*az*dt*dt ;
        noisy_state.vx += ax*dt ;
        noisy_state.vy += ay*dt ;
        noisy_state.vz += az*dt ;
        return noisy_state ;
    }

    double std_ax() { return std_ax_ ; }
    double std_ay() { return std_ay_ ; }
    double std_az() { return std_az_ ; }

protected:
    double std_ax_ ;
    double std_ay_ ;
    double std_az_ ;
};

#endif // LINEARCVMOTIONMODEL3D_CUH
