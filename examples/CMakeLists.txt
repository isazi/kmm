add_executable(vector_add ${PROJECT_SOURCE_DIR}/examples/vector_add.cu)
target_compile_features(vector_add PRIVATE cxx_std_17)
target_link_libraries(vector_add PRIVATE kmm)

add_executable(point_in_poly ${PROJECT_SOURCE_DIR}/examples/point_in_poly.cu)
target_compile_features(point_in_poly PRIVATE cxx_std_17)
target_link_libraries(point_in_poly PRIVATE kmm)

add_executable(matrix_multiply ${PROJECT_SOURCE_DIR}/examples/matrix_multiply.cu)
target_compile_features(matrix_multiply PRIVATE cxx_std_17)
target_link_libraries(matrix_multiply PRIVATE kmm)

add_executable(histogram ${PROJECT_SOURCE_DIR}/examples/histogram.cu)
target_compile_features(histogram PRIVATE cxx_std_17)
target_link_libraries(histogram PRIVATE kmm)

add_executable(reduction ${PROJECT_SOURCE_DIR}/examples/reduction.cu)
target_compile_features(reduction PRIVATE cxx_std_17)
target_link_libraries(reduction PRIVATE kmm)