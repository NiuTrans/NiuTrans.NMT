# CMake minimum version
cmake_minimum_required(VERSION 2.8)

# Project's name
project(NiuTrans.NMT)

# The prefix of the generated executable file
set(NIUTRANS_EXE "NiuTrans.NMT")

# Generated file path
set(EXECUTABLE_OUTPUT_PATH ../bin)

# Use CMAKE_MACOSX_RPATH for MacOS 
set(CMAKE_MACOSX_RPATH 1)

# Open floder manage
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
option(USE_CUDA "Use CUDA" OFF)
option(USE_MKL "Use MKL" OFF)
option(USE_OPENBLAS "Use OpenBLAS" OFF)
option(USE_FP16 "Use FP16" OFF)

if (USE_CUDA)
    message(STATUS "${CUDA_TOOLKIT_ROOT_DIR}")
endif()
if(USE_MKL)
    message(STATUS "${INTEL_ROOT}")
endif()
if(USE_OPENBLAS)
    message(STATUS "${OPENBLAS_ROOT}")
endif()

# Find all the .cpp .h .cu .chu files in source folder
file(GLOB_RECURSE CPP_FILES source/*.cpp)
file(GLOB_RECURSE H_FILES source/*.h)
file(GLOB_RECURSE CU_FILES source/*.cu)
file(GLOB_RECURSE CUH_FILES source/*.cuh)

function(assign_source_group)
    foreach(_source IN ITEMS ${ARGN})
        if (IS_ABSOLUTE "${_source}")
            file(RELATIVE_PATH _source_rel "${CMAKE_CURRENT_SOURCE_DIR}" "${_source}")
        else()
            set(_source_rel "${_source}")
        endif()
        get_filename_component(_source_path "${_source_rel}" PATH)
        string(REPLACE "/" "\\" _source_path_msvc "${_source_path}")
        source_group("${_source_path_msvc}" FILES "${_source}")
    endforeach()
endfunction(assign_source_group)

function(my_add_executable)
    foreach(_source IN ITEMS ${ARGN})
        assign_source_group(${_source})
    endforeach()
    if(USE_CUDA)
        cuda_add_executable(${ARGV})
    else()
        add_executable(${ARGV})
    endif()
endfunction(my_add_executable)

# Set libs and compiler options for CUDA
if(USE_CUDA)
    add_definitions(-DUSE_CUDA)
    if(USE_FP16)
        add_definitions(-DHALF_PRECISION)
    endif()
    find_package(CUDA ${CUDA_VERSION} REQUIRED)
    if(WIN32)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4819")
        set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-maxrregcount=0 -m64 --disable-warnings -use_fast_math -DUSE_CUDA")
        set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} -arch=sm_30 
            -gencode=arch=compute_30,code=sm_30 
            -gencode=arch=compute_50,code=sm_50 
            -gencode=arch=compute_52,code=sm_52 
            -gencode=arch=compute_60,code=sm_60 
            -gencode=arch=compute_61,code=sm_61 
            -gencode=arch=compute_62,code=sm_62 
            -gencode=arch=compute_70,code=sm_70 
            -gencode=arch=compute_70,code=compute_70 
        )
        set(CMAKE_POLICY_DEFAULT_CMP0028 NEW)
        link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib/x64")
        include_directories("${CUDA_TOOLKIT_ROOT_DIR}/include")
        set(CUDA_LIB_DIR "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/")
        set(CUDA_LIB_PATH ${CUDA_LIB_PATH} "${CUDA_LIB_DIR}cublas.lib")
        set(CUDA_LIB_PATH ${CUDA_LIB_PATH} "${CUDA_LIB_DIR}npps.lib")
        set(CUDA_LIB_PATH ${CUDA_LIB_PATH} "${CUDA_LIB_DIR}nppc.lib")
        set(CUDA_LIB_PATH ${CUDA_LIB_PATH} "${CUDA_LIB_DIR}cudadevrt.lib")
        set(CUDA_LIB_PATH ${CUDA_LIB_PATH} "${CUDA_LIB_DIR}curand.lib")
    else()
        set(CMAKE_CXX_FLAGS "-fPIC -msse4.2 -w -march=native -Wno-enum-compare -Wno-sign-compare -Wno-format -Wno-dev -O3 -DNDEBUG -rdynamic")
        if(USE_FP16)
            set(CUDA_NVCC_FLAGS "-Xcompiler -fPIC -maxrregcount=0 --disable-warnings -use_fast_math -DUSE_CUDA -DHALF_PRECISION -Wno-deprecated-gpu-targets -std=c++11 ")
            set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_60
                -gencode=arch=compute_60,code=sm_60 
                -gencode=arch=compute_61,code=sm_61 
                -gencode=arch=compute_62,code=sm_62 
                -gencode=arch=compute_70,code=sm_70 
                -gencode=arch=compute_70,code=compute_70 
            )
        else()
            set(CUDA_NVCC_FLAGS "-Xcompiler -fPIC -maxrregcount=0 --disable-warnings -use_fast_math -DUSE_CUDA -Wno-deprecated-gpu-targets -std=c++11 ")
            set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_30 
                -gencode=arch=compute_30,code=sm_30 
                -gencode=arch=compute_50,code=sm_50 
                -gencode=arch=compute_52,code=sm_52 
                -gencode=arch=compute_60,code=sm_60 
                -gencode=arch=compute_61,code=sm_61 
                -gencode=arch=compute_62,code=sm_62 
                -gencode=arch=compute_70,code=sm_70 
                -gencode=arch=compute_70,code=compute_70 
            )
        endif()
        link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
        include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
        set(CUDA_LIB_DIR "${CUDA_TOOLKIT_ROOT_DIR}/lib64/")
        set(CUDA_LIB_PATH ${CUDA_LIB_PATH} "${CUDA_LIB_DIR}libcublas_static.a")
        set(CUDA_LIB_PATH ${CUDA_LIB_PATH} "${CUDA_LIB_DIR}libculibos.a")
        set(CUDA_LIB_PATH ${CUDA_LIB_PATH} "${CUDA_LIB_DIR}libnpps_static.a")
        set(CUDA_LIB_PATH ${CUDA_LIB_PATH} "${CUDA_LIB_DIR}libnppc_static.a")
        set(CUDA_LIB_PATH ${CUDA_LIB_PATH} "${CUDA_LIB_DIR}libcudadevrt.a")
        set(CUDA_LIB_PATH ${CUDA_LIB_PATH} "${CUDA_LIB_DIR}libcurand_static.a")
        set(CUDA_LIB_PATH ${CUDA_LIB_PATH} "/usr/lib64/libdl.so.2")
    endif()
endif()

# Set libs and compiler options for MKL
if(USE_MKL)
    add_definitions(-DUSE_BLAS -DMKL)
    set(COMPILER_DIR "${INTEL_ROOT}/compiler")
    set(MKL_DIR "${INTEL_ROOT}/mkl")
    set(CPU_ARCH intel64)
    if(WIN32)
        set(CMAKE_CXX_FLAGS "-DNDEBUG -DUSE_BLAS -DMKL")
        link_directories(${MKL_DIR}/lib/intel64/)
        link_directories(${COMPILER_DIR}/lib/intel64)
        include_directories(${MKL_DIR}/include)
        set(COMPILER_LIB_DIR "${COMPILER_DIR}/lib/intel64/")
        set(MKL_LIB_DIR "${MKL_DIR}/lib/intel64/")
        set(MKL_LIB_PATH ${MKL_LIB_PATH} "${MKL_LIB_DIR}mkl_intel_lp64.lib")
        set(MKL_LIB_PATH ${MKL_LIB_PATH} "${MKL_LIB_DIR}mkl_core.lib")
        set(MKL_LIB_PATH ${MKL_LIB_PATH} "${MKL_LIB_DIR}mkl_intel_thread.lib")
        set(MKL_LIB_PATH ${MKL_LIB_PATH} "${COMPILER_LIB_DIR}libiomp5md.lib")
    else()
        set(CMAKE_CXX_FLAGS "-std=c++11 -msse4.2 -w -march=native -Wno-enum-compare -Wno-sign-compare -Wno-reorder -Wno-format -O3 -flto -DNDEBUG -rdynamic -fkeep-inline-functions -fPIC -DUSE_BLAS -DMKL")
        link_directories(${MKL_DIR}/lib/intel64/)
        link_directories(${COMPILER_DIR}/lib/intel64)
        include_directories(${MKL_DIR}/include)
        set(COMPILER_LIB_DIR "${COMPILER_DIR}/lib/intel64/")
        set(MKL_LIB_DIR "${MKL_DIR}/lib/intel64/")
        set(MKL_LIB_PATH ${MKL_LIB_PATH} "${MKL_LIB_DIR}libmkl_intel_lp64.a")
        set(MKL_LIB_PATH ${MKL_LIB_PATH} "${MKL_LIB_DIR}libmkl_core.a")
        set(MKL_LIB_PATH ${MKL_LIB_PATH} "${MKL_LIB_DIR}libmkl_intel_thread.a")
        set(MKL_LIB_PATH ${MKL_LIB_PATH} "${COMPILER_LIB_DIR}libiomp5.a")
    endif()
endif()

# Set libs and compiler options for OpenBLAS
if(USE_OPENBLAS)
    add_definitions(-DUSE_BLAS -DMKL)
    set(OPENBLAS_INCLUDE_DIR "${OPENBLAS_ROOT}/include")
    set(OPENBLAS_LIB_DIR "${OPENBLAS_ROOT}/lib")
    if(WIN32)
        set(CMAKE_CXX_FLAGS "-DNDEBUG -DUSE_BLAS -DOPENBLAS")
        link_directories(${OPENBLAS_LIB_DIR})
        include_directories(${OPENBLAS_INCLUDE_DIR})
        set(OPENBLAS_LIB_PATH ${OPENBLAS_LIB_PATH} "${OPENBLAS_LIB_DIR}/libopenblas.lib")
    else()
        set(CMAKE_CXX_FLAGS "-std=c++11 -msse4.2 -w -march=native -Wno-enum-compare -Wno-sign-compare -Wno-reorder -Wno-format -O3 -flto -DNDEBUG -rdynamic -fkeep-inline-functions -fPIC -DUSE_BLAS -DOPENBLAS")
        message( STATUS "BLAS found: ${BLAS_LIBRARIES}")
        link_directories(${OPENBLAS_LIB_DIR})
        include_directories(${OPENBLAS_INCLUDE_DIR})
        set(OPENBLAS_LIB_PATH ${OPENBLAS_LIB_PATH} "${OPENBLAS_LIB_DIR}/libopenblas.a")
        set(OPENBLAS_LIB_PATH ${OPENBLAS_LIB_PATH} "${OPENBLAS_LIB_DIR}/libopenblas_haswellp-r0.3.10.a")
        message(STATUS "OpenBLAS Lib: ${OPENBLAS_LIB_PATH}")
    endif()
endif()

# Integrate all libs
set(CUDA_LIB ${CUDA_LIB_PATH})
set(MKL_LIB ${MKL_LIB_PATH})
set(OPENBLAS_LIB ${OPENBLAS_LIB_PATH})

# Add executable files to project
if(USE_CUDA)
    set(NIUTRANS_EXE "${NIUTRANS_EXE}")
    my_add_executable(${NIUTRANS_EXE} ${CPP_FILES} ${H_FILES} ${CU_FILES} ${CUH_FILES})
else()
    set(NIUTRANS_EXE "${NIUTRANS_EXE}")
    my_add_executable(${NIUTRANS_EXE} ${CPP_FILES} ${H_FILES})
endif()

# Link libs to executable files
if(WIN32)
    add_definitions(-DWIN32)
    if(USE_CUDA)
        message(STATUS "On Windows, use CUDA")
        target_link_libraries(${NIUTRANS_EXE} ${CUDA_LIB})
    elseif(USE_MKL)
        message(STATUS "On Windows, use MKL")
        target_link_libraries(${NIUTRANS_EXE} ${MKL_LIB})
    elseif(USE_OPENBLAS)
        message(STATUS "On Windows, use OpenBLAS")
        target_link_libraries(${NIUTRANS_EXE} ${OPENBLAS_LIB})
    else()
        message(STATUS "On Windows")
    endif()
else()
    add_definitions(-std=c++11)
    if(USE_CUDA)
        message(STATUS "On Linux, use CUDA")
        target_link_libraries(${NIUTRANS_EXE} ${CUDA_LIB} -lpthread -lcudart -lnvidia-ml)
    elseif(USE_MKL)
        message(STATUS "On Linux, use MKL")
        target_link_libraries(${NIUTRANS_EXE} ${MKL_LIB}  -lpthread -liomp5 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -ldl)
    elseif(USE_OPENBLAS)
        message(STATUS "On Linux, use OpenBLAS")
        target_link_libraries(${NIUTRANS_EXE} ${OPENBLAS_LIB} -lpthread -lopenblas)
    else()
        set(CMAKE_CXX_FLAGS "-std=c++11 -msse4.2 -w -march=native -Wno-enum-compare -Wno-sign-compare -Wno-reorder -Wno-format -O3 -flto -DNDEBUG -rdynamic -fkeep-inline-functions -fPIC")
        message(STATUS "On Linux")
        target_link_libraries(${NIUTRANS_EXE} -lpthread)
    endif()
endif()

message(STATUS "name of executable file:" ${NIUTRANS_EXE})