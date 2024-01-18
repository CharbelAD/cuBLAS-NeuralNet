// Compile gcc -o ./ann main.c matrix.c ann.c mnist.c -lm

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "cublas_utils.h"

#include "mnist.h"
#include "matrix.h"
#include "ann.h"
#include <math.h>
#include <string.h>
#include <time.h>

int idx2f(int i, int j, int ld){
    return j * ld + i;
}

void populate_minibatch(double *x, double* y, unsigned* minibatch_idx, unsigned minibatch_size, image * img, unsigned img_size, byte* label, unsigned label_size);

void zero_to_n(unsigned n, unsigned* t)
{
    for (unsigned i = 0; i < n; i++)
    {
        t[i] = i;
    }
}

void shuffle(unsigned *t, const unsigned size, const unsigned number_of_switch)
{
    zero_to_n(size, t);
    for (unsigned i = 0; i < number_of_switch; i++)
    {
        unsigned x = rand() % size;
        unsigned y = rand() % size;
        unsigned tmp = t[x];
        t[x] = t[y];
        t[y] = tmp;
    }
}

double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

double dsigmoid(double x)
{
    return sigmoid(x)*(1-sigmoid(x));
}

double accuracy_cmajr(cublasHandle_t handle, cudaStream_t stream, image* test_img, byte* test_label, unsigned datasize, unsigned minibatch_size, ann_t *nn)
{
    unsigned good = 0;
    unsigned idx[datasize];    
    double *x = (double *) malloc(28 * 28 * minibatch_size * sizeof(double));
    double *y = (double *) malloc(10 * minibatch_size * sizeof(double));

    zero_to_n(datasize, idx);
    
    for (int i = 0; i < datasize - minibatch_size; i += minibatch_size)
    {        
        populate_minibatch(nn->layers[0]->activations->m, y, &idx[i], minibatch_size, test_img, 28*28, test_label, 10);
        //memcpy(nn->layers[0]->activations->m, x, 28*28 * minibatch_size * sizeof(double));     
        
        forward(handle, stream, nn, "sigmoid");
        for (int col = 0; col < minibatch_size; col++)
        {
            int idxTrainingData = col + i;
            double max = 0;
            unsigned idx_max = 0;
            for (int row = 0; row < 10; row++)
            {
                int idx = col * 10 + row;  // Adjusted for column-major order
                if (nn->layers[nn->number_of_layers-1]->activations->m[idx] > max)
                {
                    max = nn->layers[nn->number_of_layers-1]->activations->m[idx];
                    idx_max = row;
                }
            }
            if (idx_max == test_label[idxTrainingData])
            {
                good++;
            }
        }
    }    
    free(x);
    free(y);

    unsigned ntests = (datasize/minibatch_size) * minibatch_size;
    return (100.0 * (double) (good) / ntests);
}

void populate_minibatch(double * x, double * y, unsigned * minibatch_idx, unsigned minibatch_size, image * img, unsigned img_size, byte* label, unsigned label_size)
{
    for (int col = 0; col < minibatch_size; col++)
    {
        for (int row = 0; row < img_size; row++)
        {
            // Modified indexing for column-major order
            x[col * img_size + row] = (double) img[minibatch_idx[col]][row] / 255.0;
        }

        for (int row = 0; row < 10; row++)
        {
            // Modified indexing for column-major order
            y[col * 10 + row] = 0.0;
        }

        // Modified indexing for column-major order
        y[col * 10 + label[minibatch_idx[col]]] = 1.0;
    }
}


int main(int argc, char *argv[])
{
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    printf("Loading images...\n");
    fflush(stdin);
    srand(time(0));
    unsigned datasize, ntest;
    image* train_img = read_images("train-images-idx3-ubyte", &datasize);
    byte* train_label = read_labels("train-labels-idx1-ubyte", &datasize);
    image* test_img = read_images("t10k-images-idx3-ubyte", &ntest);
    byte* test_label = read_labels("t10k-labels-idx1-ubyte", &ntest);
    printf("%d train and %d test images loaded...\n", datasize, ntest);
    fflush(stdin);

    ann_t * nn;
    double alpha = 0.05;
    unsigned minibatch_size = 16;
    unsigned number_of_layers = 3;
    unsigned nneurons_per_layer[3] = {28*28, 30, 10};
    nn = create_ann(alpha, minibatch_size, number_of_layers, nneurons_per_layer);
    //print_nn(nn);

    printf("starting accuracy %lf\n", accuracy_cmajr(cublasH, stream, test_img, test_label, ntest, minibatch_size, nn));

    
    unsigned *shuffled_idx;// = (unsigned *)malloc(datasize*sizeof(unsigned));
    cudaMallocManaged(&shuffled_idx, datasize*sizeof(unsigned));
    double *x;// = (double *) malloc(28*28 * minibatch_size * sizeof( double ));
    cudaMallocManaged(&x, 28*28 * minibatch_size * sizeof( double ));
    double *y;// = (double *) malloc(10 * minibatch_size * sizeof( double ));
    cudaMallocManaged(&y, 10 * minibatch_size * sizeof( double ));
    matrix_t *out = alloc_matrix(10, minibatch_size, false);
    
    for (int epoch = 0; epoch < 2; epoch ++)
    {
        printf("start learning epoch %d\n", epoch);

        shuffle(shuffled_idx, datasize, datasize);

        for (int i = 0; i < datasize - minibatch_size ; i+= minibatch_size)
        {
            // TODO: profile the memcpy time saved
            populate_minibatch(nn->layers[0]->activations->m, y, shuffled_idx+i, minibatch_size, train_img, 28*28, train_label, 10);
            //memcpy(nn->layers[0]->activations->m, x, 28 * 28 * minibatch_size * sizeof(double));
            forward(cublasH, stream, nn, "sigmoid");
            memcpy(out->m, y, 10 * minibatch_size * sizeof(double));            
            backward(cublasH, stream, nn, out, "dsigmoid");            
        }     
        printf("epoch %d accuracy %lf\n", epoch, accuracy_cmajr(cublasH, stream, test_img, test_label, ntest, minibatch_size, nn));
    }

    cudaFree(x);
    cudaFree(y);
    cudaFree(shuffled_idx);
    destroy_matrix(out);   
    
    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}

