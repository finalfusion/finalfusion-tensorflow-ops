list(GET TF_CFG 0 LIBTENSORFLOW_FRAMEWORK)
list(GET TF_CFG 1 TF_INCLUDE)

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -DNDEBUG")
set(LIBTENSORFLOW_FRAMEWORK "${LIBTENSORFLOW_FRAMEWORK}/libtensorflow_framework.so.1")

add_library(libtensorflow_framework SHARED IMPORTED GLOBAL)
set_target_properties(libtensorflow_framework PROPERTIES
        IMPORTED_LOCATION ${LIBTENSORFLOW_FRAMEWORK}
        INTERFACE_INCLUDE_DIRECTORIES ${TF_INCLUDE})

add_library(finalfusion_tf lookup/ff_lookup.cc)
target_include_directories(finalfusion_tf PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(finalfusion_tf finalfusion_cxx libtensorflow_framework)

add_library(finalfusion_tf_op SHARED kernel/ff_lookup_kernels.cc ops/ff_lookup_ops.cc)
target_include_directories(finalfusion_tf_op PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(finalfusion_tf_op finalfusion_tf)
