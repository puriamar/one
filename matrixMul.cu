//This program computes Matrix Multiplication on the GPU using CUDA

#include <cstdlib>
#include <cassert>
#include <iostream>

using namespace std;

__global__ matrixMul(int *a, int *b, int* c, int N){

    //Calculate global row and column for each thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    //boundary check for matrix
    if(row < N && col < N){
        int tmp = 0;
        for(int i = 0; i < N; i++)
        {
            tmp += a[row * N + i] * b[i * N + col];
        }   
    }

    //Write Back the result
    c[row * N + col] = tmp;


}

//Initialize matrix of size N * N
void init_matrix(int *m, int N)
{
    for(int i = 0; i < N; i++){
        m[i] = rand() % 100; 
    }
}

//CPU calculation
void cpu(int *a, int *b, int *c, int N)
{
    int tmp;
    //each row
    for(int i = 0; i < N; i++)
    {
        //each col
        for (int j = 0; j < N; j++)
        {
            //each row-col pair
            tmp = 0;
            for (int k = 0; k < N; k++)
            {
                tmp += a[i * N + k] * b[k * N + j];
            }

            //check each result
            assert(tmp == c[i * N + j]);
        }
    }

}

int main(){
    //matrix dimensin, square matrix 2 ^ 10 * 2 ^ 10
    int N = 1 << 10;

    size_t  bytes = N * N * sizeof(int);

    //Allocate memory for matrices
    int *a, int *b, int *c;
    
    cudaMallocManaged(&a, bytes); //input matrix of N * N size
    cudaMallocManaged(&b, bytes); // input matrix of N * N size
    cudaMallocManaged(&c, bytes); // output matrix of N * N size

    //Initialize our matrices
    init_matrix(a, N);
    init_matrix(b, N);

    // Set our block and Grid dimension

    int threads = 16;
    int blocks = (N + threads - 1) / threads;

    //set up kernel launch parameters
    dim3 THREADS(threads, threads);
    dim3 BLOCKS(blocks, blocks);

    //launch our kernel
    matrixMul<<<BLOCKS, THREADS>>>(a, b, c, N);

    cudaDeviceSynchronize();

    //verify result
    cpu(a, b, c, N);

    cout << "Program completed successfully" << endl;
    return 0;
}