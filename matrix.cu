#include "matrix.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "cublas_utils.h"
#include <stdlib.h>
#include <string.h>


#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define THREADS_PER_BLOCK 512

matrix_t * alloc_matrix(unsigned rows, unsigned columns, bool zero)
{
    //matrix_t * res;
    //CUDA_CHECK(cudaMallocManaged((void **) &res, sizeof(matrix_t)));
    matrix_t * res = (matrix_t*) malloc( sizeof(matrix_t) );
    CUDA_CHECK(cudaMallocManaged((void **) &(res->m), columns * rows * sizeof(double)));
    if (zero) CUDA_CHECK(cudaMemset(res->m, 0, columns * rows * sizeof(double)));  // https://forums.developer.nvidia.com/t/can-i-set-a-floats-to-zero-with-cudamemset/153706
    res->columns = columns;
    res->rows = rows;
    return res;
}

matrix_t * alloc_matrix_device(unsigned rows, unsigned columns, bool zero)
{
    matrix_t * res = (matrix_t*) malloc( sizeof(matrix_t) );
    CUDA_CHECK(cudaMalloc((void **) &(res->m), columns * rows * sizeof(double)));
    if (zero) CUDA_CHECK(cudaMemset(res->m, 0, columns * rows * sizeof(double)));  // https://forums.developer.nvidia.com/t/can-i-set-a-floats-to-zero-with-cudamemset/153706
    res->columns = columns;
    res->rows = rows;
    return res;
}

__global__ void set_one(double* vec, int n){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < n)
		vec[index] = 1.0;
}


matrix_t * alloc_ones_device(cudaStream_t stream, unsigned rows, unsigned columns)
{
    // matrix_t * res;
    // CUDA_CHECK(cudaMallocManaged((void **) &res, sizeof(matrix_t)));
    matrix_t * res = (matrix_t*) malloc( sizeof(matrix_t) );
    CUDA_CHECK(cudaMalloc((void **) &(res->m), columns * rows * sizeof(double)));
    int n = rows * columns;
    set_one<<< (n + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, stream>>>(res->m, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    res->columns = columns;
    res->rows = rows;
    return res;
}

void destroy_matrix(matrix_t *m)
{
    //printf("free %p %p\n", m, m->m);
    cudaFree(m->m);
    free(m);
}

// TODO: review, may need to go to col-major
void print_matrix(matrix_t *m, bool is_short){
    unsigned lim_rows = 0;
    unsigned lim_col = 0;

    if (is_short)
    {
        lim_rows = MIN(m->rows, 4);
        lim_col = MIN(m->columns, 10);
    }
    else
    {
        lim_rows = m->rows;
        lim_col = m->columns;
    }

    for (int row = 0; row < lim_rows; row ++)
    {
        for (int col = 0; col < lim_col; col ++)
        {
            printf("%.2lf ", m->m[col + row * m->columns]);
        }
        if (is_short && lim_col != m->columns) printf("...");
        printf("\n");
    }
    if (is_short && lim_rows != m->rows) printf("...\n");
}


__global__ void hadamard_prod(double *v1, double *v2, double *res, int n){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < n)
		res[index] = v1[index] * v2[index];
}


void hadamard_product(cudaStream_t stream, matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)   &&
             (m1->columns == res->columns)  &&
             (m1->rows == m2->rows)         &&
             (m1->rows == res->rows));

    // for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    // {
    //         res->m[idx] = m1->m[idx] * m2->m[idx];
    // }
    int n = m1->rows * m1->columns;
    hadamard_prod<<< (n + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, stream>>>(m1->m, m2->m, res->m, n);
    //CUDA_CHECK(cudaDeviceSynchronize());
}

void cumatrix_add(cublasHandle_t handle, matrix_t *m1, matrix_t *m2, matrix_t *res, double* beta)
{
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    double alpha = 1.0;
    int m = m1->rows;
    int n = m2->columns;
    int lda = m1->rows;
    int ldb = m2->rows;
    int ldc = res->rows;

    // printf("%d, %d, %d, %d, %d\n", m, n, lda, ldb, ldc);
    // printf("(%d, %d),(%d, %d),(%d, %d)\n",  m1->rows, m1->columns, m2->rows, m2->columns, res->rows, res->columns);
    CUBLAS_CHECK(cublasDgeam(handle, transa, transb, m, n, &alpha, m1->m, lda, beta, m2->m, ldb, res->m, ldc));
}

void matrix_sum(cublasHandle_t handle, matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)  &&
             (m1->columns == res->columns) &&
             (m1->rows == m2->rows)        &&
             (m1->rows == res->rows));

    double beta = 1.0;
    cumatrix_add(handle, m1, m2, res, &beta);

    //CUDA_CHECK(cudaDeviceSynchronize());
}

void matrix_minus(cublasHandle_t handle, matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)  &&
             (m1->columns == res->columns) &&
             (m1->rows == m2->rows)        &&
             (m1->rows == res->rows));

    double beta = -1.0;
    cumatrix_add(handle, m1, m2, res, &beta);

    //CUDA_CHECK(cudaDeviceSynchronize());
}

void matrix_mul(cublasHandle_t handle, matrix_t *m1, matrix_t *m2, matrix_t *res, 
                bool transposea, bool transposeb, double alpha)
{
    // TODO: rewrite assertions to account for transpositions
    // assert ( (m1->columns == m2->rows)  &&
    //          (m1->rows == res->rows)    &&
    //          (m2->columns == res->columns));

    cublasOperation_t transa = transposea ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transb = transposeb ? CUBLAS_OP_T : CUBLAS_OP_N;
    double beta = 0;
    // m, n, and k are tied to the mathematical dimensions of Op(A) and Op(B), not with their physical representation
    int m = transposea ? m1->columns : m1->rows;
    int n = transposeb ? m2->rows : m2->columns;
    int k = transposea ? m1->rows : m1->columns;
    // lda, ldb, and ldc are tied to the way the matrices are physically stored, they don't change with Op(A) and Op(b)
    int lda = m1->rows;
    int ldb = m2->rows;
    int ldc = res->rows;

    // printf("%d, %d, %d, %d, %d, %d\n", m, n, k, lda, ldb, ldc);
    // printf("(%d, %d),(%d, %d),(%d, %d), A^T %d, B^T %d\n",  m1->rows, m1->columns, m2->rows, m2->columns, res->rows, res->columns, transposea, transposeb);
    // fflush(stdout);
    CUBLAS_CHECK(cublasDgemm(handle, transa, transb, m, n, k, &alpha, m1->m, lda, m2->m, ldb, &beta, res->m, ldc));
    //CUDA_CHECK(cudaDeviceSynchronize());
}

template<class F>
__global__ void matrix_apply(double*  m1, F f, double *res, int n){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n)
        res[index] = f(m1[index]);
}

__device__ double sigmoid_d(double x)
{
    return 1 / (1 + exp(-x));
}

__global__ void sigmoid_kernel(double*  m1, double *res, int n){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n)
        res[index] = sigmoid_d(m1[index]);
}

__device__ double dsigmoid_d(double x)
{
    return sigmoid_d(x)*(1-sigmoid_d(x));
}

__global__ void dsigmoid_kernel(double*  m1, double *res, int n){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n)
        res[index] = dsigmoid_d(m1[index]);
}

void matrix_function(cudaStream_t stream, matrix_t *m1, const char* fct, matrix_t *res)
{
    assert ( (m1->columns == res->columns) &&             
             (m1->rows == res->rows));

    int n = m1->rows * m1->columns;
    if (strcmp(fct, "sigmoid") == 0){
        sigmoid_kernel<<<(n + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, stream>>>(m1->m, res->m, n);
    }
    else if (strcmp(fct, "dsigmoid") == 0){
        dsigmoid_kernel<<<(n + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, stream>>>(m1->m, res->m, n);
    }
    //CUDA_CHECK(cudaDeviceSynchronize());
}

// NOTE: Unused and should be avoided. Prefer transposing while doing other operations if possible
void matrix_transpose(cublasHandle_t handle, matrix_t *m1, matrix_t *res)
{
    assert ( (m1->columns == res->rows) &&             
             (m1->rows == res->columns));

    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;
    double alpha = 1.0, beta = 0.0;
    int m = res->rows;
    int n = res->columns;
    int lda = m1->rows;
    int ldb = res->rows;  // Doesn't matter
    int ldc = res->rows;

    // printf("%d, %d, %d, %d, %d\n", m, n, lda, ldb, ldc);
    // printf("(%d, %d),(%d, %d),(%d, %d)\n",  m1->rows, m1->columns, m2->rows, m2->columns, res->rows, res->columns);
    CUBLAS_CHECK(cublasDgeam(handle, transa, transb, m, n, &alpha, m1->m, lda, &beta, NULL, ldb, res->m, ldc));
}

void matrix_scalar(cublasHandle_t handle, matrix_t *m1, double alpha)
{
    int n = m1->rows * m1->columns;
    int incx = 1;
    CUBLAS_CHECK(cublasDscal(handle, n, &alpha, m1->m, incx));
}

// NOTE: currently unused
void matrix_copy(matrix_t *dest, const matrix_t *src)
{
    assert ( (dest->rows == src->rows)      &&             
             (dest->columns == src->columns));

    cudaDeviceSynchronize();
    memcpy(dest->m, src->m, src->columns * src->rows * sizeof(double));
    cudaDeviceSynchronize();
}