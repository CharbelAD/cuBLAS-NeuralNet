#include "matrix.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "cublas_utils.h"
#include <stdlib.h>
#include <string.h>


#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

matrix_t * alloc_matrix(unsigned rows, unsigned columns)
{
    matrix_t * res;
    CUDA_CHECK(cudaMallocManaged((void **) &res, sizeof(matrix_t)));
    CUDA_CHECK(cudaMallocManaged((void **) &(res->m), columns * rows * sizeof(double)));
    CUDA_CHECK(cudaMemset(res->m, 0, columns * rows * sizeof(double)));  // https://forums.developer.nvidia.com/t/can-i-set-a-floats-to-zero-with-cudamemset/153706
    res->columns = columns;
    res->rows = rows;
    return res;
}

void destroy_matrix(matrix_t *m)
{
    //printf("free %p %p\n", m, m->m);
    cudaFree(m->m);
    cudaFree(m);
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

// TODO: make a kernel
void hadamard_product(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)   &&
             (m1->columns == res->columns)  &&
             (m1->rows == m2->rows)         &&
             (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    {
            res->m[idx] = m1->m[idx] * m2->m[idx];
    }
}

void matrix_sum(cublasHandle_t handle, matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)  &&
             (m1->columns == res->columns) &&
             (m1->rows == m2->rows)        &&
             (m1->rows == res->rows));

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    double alpha = 1.0, beta = 1.0;
    int m = m1->rows;
    int n = m2->columns;
    int lda = m1->rows;
    int ldb = m2->rows;
    int ldc = res->rows;

    // printf("%d, %d, %d, %d, %d\n", m, n, lda, ldb, ldc);
    // printf("(%d, %d),(%d, %d),(%d, %d)\n",  m1->rows, m1->columns, m2->rows, m2->columns, res->rows, res->columns);
    CUBLAS_CHECK(cublasDgeam(handle, transa, transb, m, n, &alpha, m1->m, lda, &beta, m2->m, ldb, res->m, ldc));
}

void matrix_minus(cublasHandle_t handle, matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)  &&
             (m1->columns == res->columns) &&
             (m1->rows == m2->rows)        &&
             (m1->rows == res->rows));
             
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    double alpha = 1.0, beta = -1.0;
    int m = m1->rows;
    int n = m2->columns;
    int lda = m1->rows;
    int ldb = m2->rows;
    int ldc = res->rows;

    // printf("%d, %d, %d, %d, %d\n", m, n, lda, ldb, ldc);
    // printf("(%d, %d),(%d, %d),(%d, %d)\n",  m1->rows, m1->columns, m2->rows, m2->columns, res->rows, res->columns);
    CUBLAS_CHECK(cublasDgeam(handle, transa, transb, m, n, &alpha, m1->m, lda, &beta, m2->m, ldb, res->m, ldc));
}

void matrix_mul(cublasHandle_t handle, matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->rows)  &&
             (m1->rows == res->rows)    &&
             (m2->columns == res->columns));

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    double alpha = 1.0, beta = 1.0;
    int m = m1->rows;
    int n = m2->columns;
    int k = m1->columns;
    int lda = m1->rows;
    int ldb = m2->rows;
    int ldc = res->rows;

    // printf("%d, %d, %d, %d, %d, %d\n", m, n, k, lda, ldb, ldc);
    // printf("(%d, %d),(%d, %d),(%d, %d)\n",  m1->rows, m1->columns, m2->rows, m2->columns, res->rows, res->columns);
    // fflush(stdout);
    CUBLAS_CHECK(cublasDgemm(handle, transa, transb, m, n, k, &alpha, m1->m, lda, m2->m, ldb, &beta, res->m, ldc));
}

void matrix_function(matrix_t *m1, double (*f)(double), matrix_t *res)
{
    assert ( (m1->columns == res->columns) &&             
             (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    {
        res->m[idx] = f(m1->m[idx]);
    }
}

// TODO: remove and replace usages
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