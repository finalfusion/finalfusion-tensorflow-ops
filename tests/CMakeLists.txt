include(CTest)

file(COPY testdata/test.fifu DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/data)
message(${CMAKE_CURRENT_BINARY_DIR})

add_test(NAME python-init-close
        COMMAND pytest ${CMAKE_CURRENT_SOURCE_DIR}
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
        )
