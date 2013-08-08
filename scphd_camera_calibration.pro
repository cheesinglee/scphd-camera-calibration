TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
    src/main.cpp \
    src/scphdcameracalibration.cu \
    src/kernels.cu \
    src/disparitymeasurementmodel.cu \
    src/device_math.cu \
    src/lu_decomposition.cu \
    src/thrust_operators.cu

HEADERS += \
    src/types.h \
    src/disparitymeasurementmodel.cuh \
    src/linearcvmotionmodel3d.cuh \
    src/device_math.cuh \
    src/kernels.cuh \
    src/scphdcameracalibration.cuh \
    src/OrientedLinearCVMotionModel3D.cuh \
    src/lu_decomposition.cuh

QMAKE_CC = gcc
QMAKE_CXX = g++
QMAKE_CXXFLAGS += -DDEBUG
QMAKE_CXXFLAGS += -O0 -g
QMAKE_CXXFLAGS += -Wall -Wno-deprecated -fpic -DOC_NEW_STYLE_INCLUDES -fpermissive -fno-strict-aliasing

LIBS += -lconfig++ -lmatio -lhdf5

#### CUDA setup ########################################################

# Cuda sources
SOURCES -= \
    src/device_math.cu \
    src/scphdcameracalibration.cu \
    src/kernels.cu \
    src/lu_decomposition.cu \
    src/disparitymeasurementmodel.cu \
    src/thrust_operators.cu

CUDA_SOURCES += \
    src/device_math.cu \
    src/disparitymeasurementmodel.cu \
    src/kernels.cu \
    src/scphdcameracalibration.cu \
    src/lu_decomposition.cu \
    src/thrust_operators.cu


CUDA_LIBS = $$LIBS

# Path to cuda SDK install
CUDA_SDK = /home/cheesinglee/cuda-sdk

# Path to cuda toolkit install
CUDA_DIR = $$system(dirname `which nvcc`)/..

# GPU architecture
CUDA_GENCODE = -gencode=arch=compute_20,code=sm_20
#CUDA_GENCODE = -gencode=arch=compute_30,code=sm_30

# nvcc flags (ptxas option verbose is always useful)
#CUDA_GCC_BINDIR=/opt/gcc-4.3
CUDA_GCC_BINDIR=/usr/lib/nvidia-cuda-toolkit/bin
NVCCFLAGS = \
    --compiler-options -fno-strict-aliasing \
    --compiler-bindir=$$CUDA_GCC_BINDIR  \
    --ptxas-options=-O0,-v\

# include paths
#INCLUDEPATH += $$CUDA_DIR/include/cuda/
#INCLUDEPATH += $$CUDA_DIR/include/
INCLUDEPATH += $$CUDA_SDK/common/inc/
# lib dirs
#QMAKE_LIBDIR += $$CUDA_DIR/lib64
QMAKE_LIBDIR += $$CUDA_SDK/lib
QMAKE_LIBDIR += $$CUDA_SDK/common/lib
# libs
LIBS += -lcudart -lnvToolsExt
# join the includes in a line
CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')

# Prepare the extra compiler configuration (taken from the nvidia forum - i'm not an expert in this part)
cuda.input = CUDA_SOURCES
cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o

cuda.commands = $$CUDA_DIR/bin/nvcc -g -G -DTHRUST_DEBUG -DDEBUG -m64 $$CUDA_GENCODE -dc $$NVCCFLAGS $$CUDA_INC $$CUDA_LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT} \

cuda.dependcy_type = TYPE_C
cuda.depend_command = $$CUDA_DIR/bin/nvcc -g -G -M $$CUDA_INC $$NVCCFLAGS ${QMAKE_FILE_NAME}
# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_COMPILERS += cuda

QMAKE_PRE_LINK = $$CUDA_DIR/bin/nvcc $$CUDA_GENCODE -dlink $(OBJECTS) -o dlink.o $$escape_expand(\n\t)

OTHER_FILES += \
    config/calibration.cfg
