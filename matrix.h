#ifndef MATRIX_H
#define MATRIX_H
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <stdbool.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

typedef struct
{
    double * m;
    unsigned columns;
    unsigned rows;
}  matrix_t;

matrix_t * alloc_matrix(unsigned rows, unsigned columns, bool zero = false);

matrix_t * alloc_ones(unsigned rows, unsigned columns);

void destroy_matrix(matrix_t *m);

void print_matrix(matrix_t *m, bool is_short);

void hadamard_product(matrix_t *m1, matrix_t *m2, matrix_t *res);

void matrix_sum(cublasHandle_t handle, matrix_t *m1, matrix_t *m2, matrix_t *res);

void matrix_minus(cublasHandle_t handle, matrix_t *m1, matrix_t *m2, matrix_t *res);

void matrix_mul(cublasHandle_t handle, matrix_t *m1, matrix_t *m2, matrix_t *res, bool transposea=false, bool transposeb=false, double alpha=1.0);

void matrix_function(matrix_t *m1, const char * fct, matrix_t *res);

void matrix_transpose(cublasHandle_t handle, matrix_t *m1, matrix_t *res);

void matrix_scalar(cublasHandle_t handle, matrix_t *m1, double s);

void matrix_copy(matrix_t *dest, const matrix_t *src);

#endif