add_library(finalfusion_tf lookup/ff_lookup.cc)
target_include_directories(finalfusion_tf PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(finalfusion_tf finalfusion_cxx)
