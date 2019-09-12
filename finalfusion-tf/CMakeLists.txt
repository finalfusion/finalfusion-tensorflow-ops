list(GET TF_CFG 0 TF_LINK_FLAGS)
list(GET TF_CFG 1 TF_INCLUDE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG")

add_library(finalfusion_tf SHARED lookup/FFLookup.cc kernel/FFLookupKernels.cc ops/FFLookupOps.cc)
target_include_directories(finalfusion_tf PUBLIC ${PROJECT_SOURCE_DIR}/include ${TF_INCLUDE})
target_link_libraries(finalfusion_tf finalfusion_cxx ${TF_LINK_FLAGS})
