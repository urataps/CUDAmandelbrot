#include <iostream>

#define N 10

#define A {1,2,3,4,5,6,7,8,9,10}
#define B {10,9,8,7,6,5,4,3,2,1}

__global__ void add( int *a, int* b, int *c ) {
    int tid = blockIdx.x;    // handle the data at this index

    if (tid < N)
        c[tid] = a[tid] + b[tid];
}

 
int main( void ) {
    printf("Running vector addition on GPU...\n");

    int a[N] = A, b[N] = B, c[N];
    int *dev_a, *dev_b, *dev_c;

    cudaMalloc( (void**)&dev_a, N * sizeof(int) ) ;
    cudaMalloc( (void**)&dev_b, N * sizeof(int) ) ;
    cudaMalloc( (void**)&dev_c, N * sizeof(int) ) ;

    /// Copy the vectors a, b to dev_a, dev_b
    cudaMemcpy( dev_a, a, N * sizeof(int),cudaMemcpyHostToDevice );
    cudaMemcpy( dev_b, b, N * sizeof(int),cudaMemcpyHostToDevice );

    add<<<N,1>>>( dev_a, dev_b, dev_c ); // first <<<1>>> is number of parallel blocks
    cudaMemcpy( &c, dev_c, N * sizeof(int),cudaMemcpyDeviceToHost );

    cudaFree( dev_a );
    cudaFree( dev_b );
    cudaFree( dev_c );

    // display the results
    for (int i=0; i<N; i++) {
        printf( "%d + %d = %d\n", a[i], b[i], c[i] );
    }

    return 0;
}