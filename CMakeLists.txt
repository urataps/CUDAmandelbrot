cmake_minimum_required(VERSION 3.10)
project(CUDA CUDA)

set(CMAKE_CUDA_STANDARD 14)


add_executable(JULIA julia.cu)
add_executable(MANDELBROT mandelbrot.cu)
add_executable(ADD_VECTORS vector_add.cu)


find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} )

target_link_libraries(JULIA ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})
target_link_libraries(MANDELBROT ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})


set_target_properties(ADD_VECTORS PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(JULIA PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)


