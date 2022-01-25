//
// Created by alexe on 03.12.2021.
//

#include <iostream>
#include "common/cpu_bitmap.h" // This is from CUDA and GLUT --> https://github.com/tpoisot/CUDA-training

// 4k
// 3840
// 2160

// FullHD
// 3840
// 2160

#define DIM1 1920
#define DIM2 1080

#define COLOR_MAX 30
#define PRECISION 100

struct cuComplex {
    float   r;
    float   i;
    __device__  cuComplex( float a, float b ) : r(a), i(b)  {}

    __device__ float magnitude2( void ) { // the distance squared
        return r * r + i * i;
    }
    __device__ cuComplex operator*(const cuComplex& a) {
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    __device__ cuComplex operator+(const cuComplex& a) {
        return cuComplex(r+a.r, i+a.i);
    }
};

__device__ unsigned char mandelbrot( float real, float img, unsigned char* r, unsigned char* g, unsigned char* b ) {

    cuComplex z(0, 0);
    cuComplex c(real, img);

    for (int color = (255 * COLOR_MAX)  ; color > 0; color -= PRECISION) {
        z = z * z + c;

        *r = color;
        *g = color/2;
        *b = color/3;

        if (z.magnitude2() > 4) { // z is not in mandelbrot set
            return 1;
        }
    }
    return 0; // z is in mandelbrot set
}

__global__ void kernel( unsigned char *ptr, float scale ){
    // map from threadIdx/BlockIdx to pixel position
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;

    /** Translate position in grid to a complex number */

    /** The center of the coordinate system */
    int centerX = DIM1/2;
    int centerY = DIM2/2;

    /** Translating pixel coordinates to complex values */
    float real = - scale * (float)(centerX - x)/(centerY); // real part
    float img = - scale * (float)(centerY - y)/(centerX); // imaginary part

    unsigned char r, g, b;

    // now calculate the color at that position
    unsigned char mandelbrotValue = mandelbrot( real, img, &r, &g, &b);

    // make mandelbrot value a float
    ptr[offset*4 + 0] = mandelbrotValue * r; // R
    ptr[offset*4 + 1] = mandelbrotValue * g; // G
    ptr[offset*4 + 2] = mandelbrotValue * b; // B
}

int main(void){

    unsigned char    *dev_bitmap;
    float GPU_scale = 1.2;

    CPUBitmap bitmap( DIM1, DIM2 ); // GLUT function
    dim3 grid(DIM1, DIM2); // actually a 2 dimensional grid, api naming conventions


    // Calling the GPU
    cudaMalloc( (void**)&dev_bitmap, bitmap.image_size());
    kernel<<<grid, 1>>>(dev_bitmap, GPU_scale);

    // copying the result back to the CPU
    cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);
    bitmap.display_and_exit();

    cudaFree( dev_bitmap );
}