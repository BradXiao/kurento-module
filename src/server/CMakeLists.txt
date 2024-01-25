include(CodeGenerator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-function")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-function")


find_package(CUDAToolkit 12.1 REQUIRED)
message("CUDAToolkit_LIBRARY_DIR=${CUDAToolkit_LIBRARY_DIR}")
message("CUDAToolkit_INCLUDE_DIRS=${CUDAToolkit_INCLUDE_DIRS}")

file (GLOB YOLOV7 "${CMAKE_CURRENT_SOURCE_DIR}/implementation/objects/yolov7/*.cpp")
message("YOLOV7=${YOLOV7}")

generate_code(
  MODELS ${CMAKE_CURRENT_SOURCE_DIR}/interface
  SERVER_STUB_DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/implementation/objects

  SERVER_IMPL_LIB_EXTRA_SOURCES  ${YOLOV7}
  SERVER_IMPL_LIB_EXTRA_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/implementation/objects/yolov7
  SERVER_IMPL_LIB_EXTRA_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS}
  SERVER_IMPL_LIB_EXTRA_LIBRARIES CUDA::cudart nvinfer nvinfer_plugin 

  MODULE_EXTRA_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/implementation/objects/yolov7
  MODULE_EXTRA_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS}
  MODULE_EXTRA_LIBRARIES CUDA::cudart nvinfer nvinfer_plugin 

)
