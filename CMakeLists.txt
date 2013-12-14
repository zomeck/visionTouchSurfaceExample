project(visionTouchSurfaceTest)
cmake_minimum_required(VERSION 2.8)
aux_source_directory(. SRC_LIST)
#aux_source_directory(. SRC_LIST)
include(FindPkgConfig)
add_executable(${PROJECT_NAME} ${SRC_LIST})

set( CMAKE_CXX_FLAGS_RELEASE "-O3" )

include_directories(/usr/include/ni)
include_directories(../visionTouchSurfaceLib/vts_utils/include)
include_directories(../visionTouchSurfaceLib/vts_sensors/include)
include_directories(../visionTouchSurfaceLib/vts_ioevents/include)
include_directories(../visionTouchSurfaceLib/vts_gestures/include)
find_package(OpenCV 2)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries( ${PROJECT_NAME} /usr/lib64/libOpenNI.so ${OpenCV_LIBS} libboost_thread.so libboost_chrono.so libboost_filesystem.so /usr/lib64/libXtst.so /usr/lib64/Leap/libLeap.so
/media/dades/vibot/roboticsProject/visionTouchSurfaceLib-build/vts_utils/libvts_utils.so
/media/dades/vibot/roboticsProject/visionTouchSurfaceLib-build/vts_sensors/libvts_sensors.so
/media/dades/vibot/roboticsProject/visionTouchSurfaceLib-build/vts_ioevents/libvts_ioevents.so
/media/dades/vibot/roboticsProject/visionTouchSurfaceLib-build/vts_gestures/libvts_gestures.so
)

