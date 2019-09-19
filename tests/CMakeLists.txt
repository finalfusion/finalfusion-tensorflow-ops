include(CTest)

file(COPY testdata/test.fifu DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/data)
message(${CMAKE_CURRENT_BINARY_DIR})

add_test(NAME eager-mode
        COMMAND pytest ${CMAKE_CURRENT_SOURCE_DIR}/test_eager_mode.py
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
        )

add_test(NAME graph-mode
        COMMAND pytest ${CMAKE_CURRENT_SOURCE_DIR}/test_graph_mode.py
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
        )
