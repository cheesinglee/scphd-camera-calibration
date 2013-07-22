#include <iostream>
#include <matio.h>
#include <vector>

#include "scphdcameracalibration.cuh"

using namespace std;

void readMatfile(const char* filename){
    mat_t* matfp =  Mat_Open(filename,MAT_ACC_RDONLY) ;
    matvar_t* matvar = Mat_VarRead(matfp,"Z1") ;
    int n_steps = matvar->dims[1] ;
    for ( int k = 0 ; k < 1 ; k++ ){
        matvar_t* Z1_k = Mat_VarGetCell(matvar,k) ;
        int n = Z1_k->dims[1] ;
        vector<double> u ;
        vector<double> v ;
        double* ptr = (double*)Z1_k->data ;
        for ( int i = 0 ; i < 2*n ; i++){
            if (i % 2 == 0)
                u.push_back(ptr[i]);
            else
                v.push_back(ptr[i]);
        }
        for ( int i =0 ; i < n ; i++ ){
            cout << u[i] << " " << v[i] << endl ;
        }
    }
}

int main(int argc, char* argv[])
{
    cout << "Hello World!" << endl;
    readMatfile("disparity_sim.mat") ;
//    SCPHDCameraCalibration calibration(argv[1]) ;
    return 0;
}

