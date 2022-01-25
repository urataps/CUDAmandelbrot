//
// Created by alexe on 03.12.2021.
//

#include <iostream>
#include "common/cpu_bitmap.h" // This is from CUDA and GLUT --> https://github.com/tpoisot/CUDA-training

#define DIM1 1920
#define DIM2 1080

struct cuComplex {
    float   r;
    float   i;
    __device__  cuComplex( float a, float b ) : r(a), i(b)  {}

    __device__ float magnitude2( void ) { //
        return r * r + i * i;
    }
    __device__ cuComplex operator*(const cuComplex& a) {
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    __device__ cuComplex operator+(const cuComplex& a) {
        return cuComplex(r+a.r, i+a.i);
    }
};

__device__ int julia( int x, int y ) {
    const float scale = 1.5; // the bigger the value, the more zoomed-out the fractal is
    float jx = scale * (float)(DIM1/2 - x)/(DIM2/2);
    float jy = scale * (float)(DIM2/2 - y)/(DIM1/2);

    /** Different values for c for interesting outputs*/
//    cuComplex c(0.355, 0.355);
//    cuComplex c(0.37, 0.1);
    cuComplex c(-0.8, 0.156);

    cuComplex a(jx, jy);

    int i = 0;
    for (i=0; i<200; i++) {
        a = a * a + c;
        if (a.magnitude2() > 1000)
            return 0;
    }
    return 1;
}

__global__ void kernel( unsigned char *ptr ){
    // map from threadIdx/BlockIdx to pixel position
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x; // the point in the two-dimensional grid

    // now calculate the value at that position
    int juliaValue = julia( x, y );
    ptr[offset*4 + 0] = 255 * juliaValue;
    ptr[offset*4 + 1] = 0;
    ptr[offset*4 + 2] = 0;
    ptr[offset*4 + 3] = 255;
}

int main(void){
    CPUBitmap bitmap( DIM1, DIM2 ); // GLUT function

    unsigned char    *dev_bitmap;
    cudaMalloc( (void**)&dev_bitmap, bitmap.image_size() );
    dim3 grid(DIM1,DIM2); // actually a 2 dimensional grid, api naming conventions

    // GPU computation
    kernel<<<grid,1>>>( dev_bitmap );

    // copying the result to the CPU
    cudaMemcpy( bitmap.get_ptr(),dev_bitmap,bitmap.image_size(),cudaMemcpyDeviceToHost );

    bitmap.display_and_exit();
    cudaFree( dev_bitmap );
}