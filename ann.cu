#include "ann.h"
#include "matrix.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "cublas_utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <stdbool.h>
#include <stdint.h>
#include <random>

double normalRand(double mu, double sigma);
//void init_weight(matrix_t* w, unsigned nneurones_prev);
void print_layer(layer_t *layer);

// double normalRand(double mu, double sigma)
// {
// 	const double epsilon = DBL_MIN;
// 	const double two_pi = 2.0*M_PI;
//     bool generate;
//     double z1;

// 	generate = !generate;

// 	if (!generate)
// 	   return z1 * sigma + mu;

// 	double u1, u2;
// 	do
// 	 {
// 	   u1 = (double) rand() / RAND_MAX;
// 	   u2 = (double) rand() / RAND_MAX;
// 	 }
// 	while ( u1 <= epsilon );

// 	double z0;
// 	z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
// 	z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
// 	return z0 * sigma + mu;
// }

// void init_weight(matrix_t* w, unsigned nneurones_prev)
// {
//     for (int idx = 0; idx < w->columns * w->rows; idx ++)
//     {
//         w->m[idx] = normalRand(0, 1 / sqrt(nneurones_prev));
//     }
// }

void init_weight(matrix_t* w, unsigned nneurones_prev, unsigned seed = 42)
{
    // Initialize the random number generator with a fixed seed
    std::mt19937 gen{seed};
 
    // Initialize a normal distribution with mean 0 and standard deviation 1/sqrt(nneurones_prev)
    std::normal_distribution<> d{0.0, 1.0 / sqrt(nneurones_prev)};
 
    // Fill the matrix with normally distributed random numbers
    for (int idx = 0; idx < w->columns * w->rows; ++idx)
    {
        w->m[idx] = d(gen);
    }
}


ann_t * create_ann(double alpha, unsigned minibatch_size, unsigned number_of_layers, unsigned* nneurons_per_layer)
{
    ann_t * nn = (ann_t *)malloc(sizeof(ann_t));

    nn->layers = (layer_t **)malloc(number_of_layers * sizeof(layer_t *));
    nn->number_of_layers = number_of_layers;
    nn->alpha = alpha;
    nn->minibatch_size = minibatch_size;

    nn->layers[0] = create_layer(0, nneurons_per_layer[0], minibatch_size, minibatch_size);
    for (int l = 1; l < number_of_layers; l++)
    {
        nn->layers[l] = create_layer(l, nneurons_per_layer[l], nneurons_per_layer[l-1], minibatch_size);
    }

    return nn;
}

layer_t * create_layer(unsigned layer_number, unsigned number_of_neurons, unsigned nneurons_previous_layer, unsigned minibatch_size)
{
    layer_t * layer = (layer_t*) malloc(sizeof(layer_t));

    layer->number_of_neurons = number_of_neurons;
    layer->minibatch_size = minibatch_size;    
    layer->activations = alloc_matrix(number_of_neurons, minibatch_size);
    layer->z = alloc_matrix(number_of_neurons, minibatch_size);
    layer->delta = alloc_matrix(number_of_neurons, minibatch_size);
    layer->weights = alloc_matrix(number_of_neurons, nneurons_previous_layer);    
    layer->biases = alloc_matrix(number_of_neurons, 1);

    if (layer_number > 0)
    {
        init_weight(layer->weights, nneurons_previous_layer);
    }

    return layer;
}

void set_input(ann_t *nn, matrix_t* input){
    matrix_copy(nn->layers[0]->activations, input);
}

void print_layer(layer_t *layer)
{
    printf("-- neurons:%d, minibatch size:%d\n", layer->number_of_neurons, layer->minibatch_size);

    printf(">> Weighted inputs --\n");
    print_matrix(layer->z, true);
    printf(">> Activations --\n");
    print_matrix(layer->activations, true);
    
    printf(">> Weights --\n");
    print_matrix(layer->weights, true);
    printf(">> Biases --\n");
    print_matrix(layer->biases, true);

    printf(">> Delta --\n");
    print_matrix(layer->delta, true);
    
}

void print_nn(ann_t *nn)
{
    printf("ANN -- nlayers:%d, alpha:%lf, minibatch size: %d\n", nn->number_of_layers, nn->alpha, nn->minibatch_size);
    for (int l = 0; l < nn->number_of_layers; l++)
    {
        printf("Layer %d ", l);
        print_layer(nn->layers[l]);
    }
}

void forward(cublasHandle_t handle, ann_t *nn, double (*activation_function)(double))
{
    for (int l = 1; l < nn->number_of_layers; l++)
    {
        matrix_t *z1 = alloc_matrix(nn->layers[l]->number_of_neurons, nn->minibatch_size);
        matrix_t *z2 = alloc_matrix(nn->layers[l]->number_of_neurons, nn->minibatch_size);
        matrix_t *one = alloc_matrix(1, nn->minibatch_size);
        for (int idx = 0; idx < one->columns*one->rows; idx++)
            one->m[idx] = 1.0;

        // printf("Mul call 1\n");
        matrix_mul(handle, nn->layers[l]->weights, nn->layers[l-1]->activations, z1); // z1 <- w^l x a^(l-1)
        CUDA_CHECK(cudaDeviceSynchronize());
        // printf("Mul call 2\n");
        matrix_mul(handle, nn->layers[l]->biases, one, z2); // z2 <- b^l x 1        
        CUDA_CHECK(cudaDeviceSynchronize());
        // printf("Sum\n");
        CUDA_CHECK(cudaDeviceSynchronize());
        matrix_sum(handle, z1, z2, nn->layers[l]->z); // z^l <- z1 + z2 <=> z^l <- w^l x a^(l-1) + b^l x 1      
        CUDA_CHECK(cudaDeviceSynchronize());
        matrix_function(nn->layers[l]->z, activation_function, nn->layers[l]->activations); // a^l = f(z^l)

        CUDA_CHECK(cudaDeviceSynchronize());
        destroy_matrix(z1);
        destroy_matrix(z2);
        destroy_matrix(one);
    }
}

void backward(cublasHandle_t handle, ann_t *nn, matrix_t *y, double (*derivative_actfunct)(double))
{
    unsigned L = nn->number_of_layers-1;

    matrix_t *dfzL = alloc_matrix(nn->layers[L]->number_of_neurons, nn->minibatch_size);

    matrix_minus(handle, nn->layers[L]->activations, y, nn->layers[L]->delta);  // delta^(L) = (a^L - y)
    CUDA_CHECK(cudaDeviceSynchronize());
    matrix_function(nn->layers[L]->z, derivative_actfunct, dfzL); // f'(z^(L))
    hadamard_product(nn->layers[L]->delta, dfzL, nn->layers[L]->delta); // delta^(L) = (a^L - y) o f'(z^(L))

    destroy_matrix(dfzL);

    for (int l = L; l > 1; l--)
    {
        matrix_t *tw, *delta_tmp, *dfz;
        tw = alloc_matrix(nn->layers[l-1]->number_of_neurons, nn->layers[l]->number_of_neurons);
        delta_tmp = alloc_matrix(nn->layers[l-1]->number_of_neurons, nn->minibatch_size);
        dfz = alloc_matrix(nn->layers[l-1]->number_of_neurons, nn->minibatch_size);

        matrix_transpose(handle, nn->layers[l]->weights, tw); // (w^l)T        
        CUDA_CHECK(cudaDeviceSynchronize());
        // printf("Mul call 3\n");
        matrix_mul(handle, tw, nn->layers[l]->delta, delta_tmp); // (w^l)T x delta^l
        CUDA_CHECK(cudaDeviceSynchronize());
        matrix_function(nn->layers[l-1]->z, derivative_actfunct, dfz); // f'(z^(l-1))
        hadamard_product(delta_tmp, dfz, nn->layers[l-1]->delta); // delta^(l-1) = (w^l)T x delta^l o f'(z^(l-1))

        CUDA_CHECK(cudaDeviceSynchronize());
        destroy_matrix(tw);
        destroy_matrix(delta_tmp);
        destroy_matrix(dfz);
    }

    for (int l = 1; l < nn->number_of_layers; l++)
    {
        matrix_t *w1, *ta;
        w1 = alloc_matrix(nn->layers[l]->number_of_neurons, nn->layers[l-1]->number_of_neurons);
        ta = alloc_matrix(nn->minibatch_size, nn->layers[l-1]->number_of_neurons);
        
        matrix_transpose(handle, nn->layers[l-1]->activations, ta); // ta <- (a^(l-1))^T
        CUDA_CHECK(cudaDeviceSynchronize());
        // printf("Mul call 4\n");
        matrix_mul(handle, nn->layers[l]->delta, ta, w1); // w1 <- delta^l x (a^(l-1))^T
        CUDA_CHECK(cudaDeviceSynchronize());
        matrix_scalar(handle, w1, nn->alpha / nn->minibatch_size); // w1 <- alpha /m . delta^l x (a^(l-1))^T
        CUDA_CHECK(cudaDeviceSynchronize());
        matrix_minus(handle, nn->layers[l]->weights, w1, nn->layers[l]->weights); // w^l <- w^l - alpha /m . delta^l x (a^(l-1))^T
        CUDA_CHECK(cudaDeviceSynchronize());

        destroy_matrix(w1);
        destroy_matrix(ta);

        matrix_t *one, *b1;
        b1 = alloc_matrix(nn->layers[l]->number_of_neurons, 1);
        one = alloc_matrix(nn->minibatch_size, 1);
        for (int idx = 0; idx < one->columns*one->rows; idx++)
            one->m[idx] = 1.0;
        // printf("Mul call 5\n");
        matrix_mul(handle, nn->layers[l]->delta, one, b1); // b1 <- delta^l x 1^T
        CUDA_CHECK(cudaDeviceSynchronize());
        matrix_scalar(handle, b1,  nn->alpha / nn->minibatch_size); // b1 <- alpha / m . delta^l x 1^T
        CUDA_CHECK(cudaDeviceSynchronize());
        matrix_minus(handle, nn->layers[l]->biases, b1, nn->layers[l]->biases); // b^l = b^l - alpha / m . delta^l x 1^T
        CUDA_CHECK(cudaDeviceSynchronize());

        destroy_matrix(one);
        destroy_matrix(b1);
    }
}