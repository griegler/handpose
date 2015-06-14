CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(handpose)


option(BUILD_POSE "build create_data/eval/train apps" ON)
option(BUILD_LIVE "link creative and pmd camera" OFF)
option(BUILD_ANNOTATION_TOOL "build annotation tool" ON)


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_BUILD_TYPE Debug)
message("CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE})



#packages
# find_package(OpenMP)
# if (OPENMP_FOUND)
#   message(STATUS "OpenMP was found.")
#   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# endif()

find_package(OpenCV REQUIRED core imgproc highgui objdetect flann)
message(STATUS "OpenCV ${OpenCV_VERSION} was found.")
include_directories(${OpenCV_INCLUDE_DIR})

find_package(Boost COMPONENTS system date_time chrono regex thread filesystem program_options timer REQUIRED)
message(STATUS "Boost was found.")
include_directories(${Boost_INCLUDE_DIR})

if(BUILD_LIVE)
  find_package(rv_cpp_utils REQUIRED CAMERA_PMD CAMERA_CREATIVE_325)
  add_definitions(-DBUILD_DATA_PROVIDER_CAM)
else()
  find_package(rv_cpp_utils REQUIRED)
endif()
message(STATUS "RV_CPP_UTILS_INCLUDE_DIRS: ${RV_CPP_UTILS_INCLUDE_DIRS}")
message(STATUS "RV_CPP_UTILS_LIBRARIES: ${RV_CPP_UTILS_LIBRARIES}")
include_directories(${RV_CPP_UTILS_INCLUDE_DIRS})

find_package(Caffe REQUIRED)
message(STATUS "CAFFE_INCLUDE_DIRS: ${CAFFE_INCLUDE_DIRS}")
message(STATUS "CAFFE_LIBRARIES: ${CAFFE_LIBRARIES}")
include_directories(${CAFFE_INCLUDE_DIRS})
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

find_package(Protobuf REQUIRED)
message(STATUS "PROTOBUF_LIBRARIES: ${PROTOBUF_LIBRARIES}")
PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS proto/pose.proto)
message(STATUS "ProtoSources: ${PROTO_SRCS}")
message(STATUS "ProtoHeaders: ${PROTO_HDRS}")
include_directories(${CMAKE_CURRENT_BINARY_DIR})


message(STATUS "${CMAKE_CURRENT_SOURCE_DIR}/include")


if(BUILD_POSE)
  add_library(pose 
    src/common.cpp
    src/create_data.cpp 
    src/data_provider_blender.cpp
    src/data_provider_csv.cpp
    src/data_provider_dtang.cpp
    src/data_provider_cam.cpp
    src/hand_patch.cpp
    src/hand_segmentation_meanshift.cpp
    src/hand_segmentation_rf.cpp
    src/hand_segmentation_threshold.cpp
    src/patch_extraction.cpp
    src/preprocess.cpp
    src/projection.cpp
    src/inference_regression_net.cpp
    src/inference_heatmap_net.cpp
    src/inference_rf_simple.cpp
    ${PROTO_SRCS}
  )
  
  set(POSE_LIBRARIES
    pose
    ${OpenCV_LIBS} 
    ${Boost_LIBRARIES} 
    ${CAFFE_LIBRARIES} 
    ${RV_CPP_UTILS_LIBRARIES} 
    ${PROTOBUF_LIBRARIES}
  )
  
  message(STATUS "POSE_LIBRARIES: ${POSE_LIBRARIES}")
  
  add_executable(create_data src/main_create_data.cpp)
  add_dependencies(create_data pose)
  target_link_libraries(create_data ${POSE_LIBRARIES})

  add_executable(eval src/main_eval.cpp)
  add_dependencies(eval pose)
  target_link_libraries(eval ${POSE_LIBRARIES})

  add_executable(nn_train src/main_nn_train.cpp)
  add_dependencies(nn_train pose)
  target_link_libraries(nn_train ${POSE_LIBRARIES})

  add_executable(rf_train src/main_rf_train.cpp)
  add_dependencies(rf_train pose)
  target_link_libraries(rf_train ${POSE_LIBRARIES})

  add_executable(seg_rf_train src/main_rf_train_segmentation.cpp)
  add_dependencies(seg_rf_train pose)
  target_link_libraries(seg_rf_train ${POSE_LIBRARIES})
    
endif()


if(BUILD_ANNOTATION_TOOL) 
    
  find_package(OpenGL REQUIRED)
  include_directories(${OPENGL_INCLUDE_DIRS})

  find_package(Qt5Core REQUIRED)
  find_package(Qt5Xml REQUIRED)
  find_package(Qt5OpenGL REQUIRED)
  find_package(Qt5Widgets REQUIRED)

  find_package(QGLVIEWER REQUIRED)
  if(QGLVIEWER_FOUND)
    message(STATUS "QGLVIEWER was found: ${QGLVIEWER_INCLUDE_DIR}")
    message(STATUS "QGLVIEWER library: ${QGLVIEWER_LIBRARIES}")
    include_directories(${QGLVIEWER_INCLUDE_DIR})
    #add_definitions(${QGLVIEWER_DEFINITIONS})
  endif(QGLVIEWER_FOUND)


  add_executable(annotation_tool 
    src/main_annotation.cpp 
    src/annotation_viewer.cpp 
    src/common.cpp 
    src/projection.cpp 
    ${PROTO_SRCS}
  )
  target_link_libraries(annotation_tool 
    Qt5::Widgets Qt5::OpenGL Qt5::Core Qt5::Xml 
    ${QGLVIEWER_LIBRARIES} 
    ${OPENGL_LIBRARIES} 
    ${OpenCV_LIBS} 
    ${Boost_LIBRARIES} 
    ${PROTOBUF_LIBRARIES}
  )

endif()



