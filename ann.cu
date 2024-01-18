#include "ann.h"
#include "matrix.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "cublas_utils.h"
#include <stdlib.h>
#include <stdio.h>
//#include <math.h>
#include <time.h>
#include <float.h>
#include <stdbool.h>
#include <stdint.h>
#include <random>
#include <vector>

double normalRand(double mu, double sigma);
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
    CUDA_CHECK(cudaDeviceSynchronize());
    return nn;
}

layer_t * create_layer(unsigned layer_number, unsigned number_of_neurons, unsigned nneurons_previous_layer, unsigned minibatch_size)
{
    layer_t * layer = (layer_t*) malloc(sizeof(layer_t));

    layer->number_of_neurons = number_of_neurons;
    layer->minibatch_size = minibatch_size;    
    layer->activations = alloc_matrix(number_of_neurons, minibatch_size, true);
    layer->z = alloc_matrix_device(number_of_neurons, minibatch_size, true);
    layer->delta = alloc_matrix_device(number_of_neurons, minibatch_size, true);
    layer->weights = alloc_matrix(number_of_neurons, nneurons_previous_layer);    
    layer->biases = alloc_matrix_device(number_of_neurons, 1, true);

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

void forward(cublasHandle_t handle, cudaStream_t stream, ann_t *nn, const char* activation_function)
{
    std::vector<matrix_t *> destroy_queue;

    for (int l = 1; l < nn->number_of_layers; l++)
    {
        matrix_t *z1 = alloc_matrix_device(nn->layers[l]->number_of_neurons, nn->minibatch_size);
        matrix_t *z2 = alloc_matrix_device(nn->layers[l]->number_of_neurons, nn->minibatch_size);
        matrix_t *one = alloc_ones_device(stream, 1, nn->minibatch_size);

        matrix_mul(handle, nn->layers[l]->weights, nn->layers[l-1]->activations, z1); // z1 <- w^l x a^(l-1)
        matrix_mul(handle, nn->layers[l]->biases, one, z2); // z2 <- b^l x 1
        matrix_sum(handle, z1, z2, nn->layers[l]->z); // z^l <- z1 + z2 <=> z^l <- w^l x a^(l-1) + b^l x 1
        matrix_function(stream, nn->layers[l]->z, activation_function, nn->layers[l]->activations); // a^l = f(z^l)

        destroy_queue.push_back(z1);
        destroy_queue.push_back(z2);
        destroy_queue.push_back(one);
    }

    for (int i = 0; i < destroy_queue.size(); ++i){
        destroy_matrix(destroy_queue[i]);
    }

    // CUDA_CHECK(cudaDeviceSynchronize());
}

void backward(cublasHandle_t handle, cudaStream_t stream, ann_t *nn, matrix_t *y, const char*derivative_actfunct)
{
    std::vector<matrix_t *> destroy_queue;
    unsigned L = nn->number_of_layers-1;

    matrix_t *dfzL = alloc_matrix_device(nn->layers[L]->number_of_neurons, nn->minibatch_size);

    matrix_minus(handle, nn->layers[L]->activations, y, nn->layers[L]->delta);  // delta^(L) = (a^L - y)
    matrix_function(stream, nn->layers[L]->z, derivative_actfunct, dfzL); // f'(z^(L))
    hadamard_product(stream, nn->layers[L]->delta, dfzL, nn->layers[L]->delta); // delta^(L) = (a^L - y) o f'(z^(L))

    destroy_queue.push_back(dfzL);

    for (int l = L; l > 1; l--)
    {
        matrix_t *delta_tmp, *dfz;
        delta_tmp = alloc_matrix_device(nn->layers[l-1]->number_of_neurons, nn->minibatch_size);
        dfz = alloc_matrix_device(nn->layers[l-1]->number_of_neurons, nn->minibatch_size);

        matrix_mul(handle, nn->layers[l]->weights, nn->layers[l]->delta, delta_tmp, true, false, 1.0); // (w^l)T x delta^l
        matrix_function(stream, nn->layers[l-1]->z, derivative_actfunct, dfz); // f'(z^(l-1))
        hadamard_product(stream, delta_tmp, dfz, nn->layers[l-1]->delta); // delta^(l-1) = (w^l)T x delta^l o f'(z^(l-1))

        destroy_queue.push_back(delta_tmp);
        destroy_queue.push_back(dfz);
    }

    for (int l = 1; l < nn->number_of_layers; l++)
    {
        matrix_t *w1;
        w1 = alloc_matrix_device(nn->layers[l]->number_of_neurons, nn->layers[l-1]->number_of_neurons);
        
        matrix_mul(handle, nn->layers[l]->delta, nn->layers[l-1]->activations, w1, false, true, nn->alpha / nn->minibatch_size); // w1 <- delta^l x (a^(l-1))^T
        matrix_minus(handle, nn->layers[l]->weights, w1, nn->layers[l]->weights); // w^l <- w^l - alpha /m . delta^l x (a^(l-1))^T

        matrix_t *one, *b1;
        b1 = alloc_matrix_device(nn->layers[l]->number_of_neurons, 1);
        one = alloc_ones_device(stream, nn->minibatch_size, 1);
        matrix_mul(handle, nn->layers[l]->delta, one, b1, false, false, nn->alpha / nn->minibatch_size); // b1 <- delta^l x 1^T
        matrix_minus(handle, nn->layers[l]->biases, b1, nn->layers[l]->biases); // b^l = b^l - alpha / m . delta^l x 1^T

        destroy_queue.push_back(w1);
        destroy_queue.push_back(one);
        destroy_queue.push_back(b1);
    }
    
    for (int i = 0; i < destroy_queue.size(); ++i){
        destroy_matrix(destroy_queue[i]);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}