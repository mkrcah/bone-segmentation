project(BONE_SEGMENTATION)

cmake_minimum_required(VERSION 2.6)


set(EXTERNAL_LIBS_SRC_DIR ${BONE_SEGMENTATION_SOURCE_DIR}/external-libs-src)


# Project dependency: ITK
FIND_PACKAGE(ITK)
IF(ITK_FOUND)
	INCLUDE(${ITK_USE_FILE})
ELSE(ITK_FOUND)
	MESSAGE(FATAL_ERROR "ITK not found. Please set ITK_DIR.")
ENDIF(ITK_FOUND)
set(ITK_LIBS ${ITK_LIBRARIES})


# Compilation of maxflow library
add_subdirectory(${EXTERNAL_LIBS_SRC_DIR}/maxflow-v3.01)

# Compilation and linking of the whole project
include_directories(${BONE_SEGMENTATION_SOURCE_DIR} filters utils 
	${EXTERNAL_LIBS_SRC_DIR}/maxflow-v3.01 
	${EXTERNAL_LIBS_SRC_DIR}/boost_1_44_0)

add_executable(SegmentBoneFromCT Main.cxx)

target_link_libraries(SegmentBoneFromCT Graph MaxFlow ${ITK_LIBS})

