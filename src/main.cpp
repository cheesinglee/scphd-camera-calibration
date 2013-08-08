#include <iostream>
#include <sstream>
#include <matio.h>
#include <vector>

#include <libconfig.h++>

#include "scphdcameracalibration.cuh"
#include "disparitymeasurementmodel.cuh"
#include "types.h"

using namespace std;

typedef struct{
    vector<double> u ;
    vector<double> v ;
} ImageMeasurement ;

vector<ImageMeasurement> readMatfile(const char* filename, const char* varname){
    mat_t* matfp =  Mat_Open(filename,MAT_ACC_RDONLY) ;

    cout << "reading " << varname << " from "  << filename << endl ;
    matvar_t* matvar = Mat_VarRead(matfp,varname) ;
    int n_steps = matvar->dims[1] ;
    cout << "n_steps = " << n_steps << endl ;
    vector<ImageMeasurement> Z(n_steps) ;
    for ( int k = 0 ; k < n_steps ; k++ ){
//        cout << "k = " << k << endl ;
        matvar_t* Z_k = Mat_VarGetCell(matvar,k) ;
        int n = Z_k->dims[1] ;
        Z[k].u.clear() ;
        Z[k].v.clear() ;
        double* ptr = (double*)Z_k->data ;
        for ( int i = 0 ; i < 2*n ; i++){
//            cout << "i = " << i << endl ;
            if (i % 2 == 0)
                Z[k].u.push_back(ptr[i]);
            else
                Z[k].v.push_back(ptr[i]);
        }
        Mat_VarFree(Z_k) ;
    }
//    cout << "free matvar " << endl ;
//    Mat_VarFree(matvar) ;

    cout << "close matfp" << endl ;
    Mat_Close(matfp) ;
    return Z ;
}



int main(int argc, char* argv[])
{
    if (argc < 3){
        cout << "Usage: scphd_camera_calibration [configuration] [data]" << endl;
        exit(0);
    }

    cout << "reading measurements" << endl ;
    vector<ImageMeasurement> Z1 = readMatfile(argv[2],"Z1") ;
    vector<ImageMeasurement> Z2 = readMatfile(argv[2],"Z2") ;

    int n_steps = Z1.size() ;
    cout << "read " << n_steps << " measurements" << endl ;

    SCPHDCameraCalibration calibration(argv[1]) ;

    calibration.writeMat("prior.mat");


    Config cfg ;
    cfg.readFile(argv[1]);
    int max_steps = cfg.lookup("max_steps") ;
    int kmax = std::min(n_steps,max_steps) ;

    for ( int k = 0 ; k < kmax ; k++){
        cout << "k = " << k << endl ;

        calibration.predict(0.5);

        if ( (k % 2) == 0)
            calibration.update(Z1[k].u,Z1[k].v,true);
        else
            calibration.update(Z2[k].u,Z2[k].v,false);
        stringstream ss ;
        ss << k << ".mat" ;
        calibration.writeMat(ss.str().data());
        calibration.checkStuff();
        calibration.resample();
    }
    return 0;
}

