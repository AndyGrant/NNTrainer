/*
  Ethereal is a UCI chess playing engine authored by Andrew Grant.
  <https://github.com/AndyGrant/Ethereal>     <andrew@grantnet.us>

  Ethereal is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Ethereal is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the

  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "trainer.h"
#include "matrix.h"
#include "vector.h"
#include "operations.h"
#include "activate.h"

int main() {

    Sample *samples = load_samples(DATAFILE, NSAMPLES);

    Network *nn = create_network(2, (Layer[]) {
        { 768, 96, &relu    , &relu_prime    },
        {  96,  1, &sigmoid , &sigmoid_prime },
    });

    Optimizer *opt  = create_optimizer(nn);
    Evaluator *eval = create_evaluator(nn);
    Gradient *grad  = create_gradient(nn);

    for (int epoch = 0; epoch < 25000; epoch++) {

        float loss = 0.0;

        for (int i = 0; i < NSAMPLES; i++) {

            sparse_evaluate_network(nn, eval, &samples[i]);
            build_backprop_grad(nn, eval, grad, &samples[i]);

            loss += loss_function(eval->activations[nn->layers-1]->values[0], samples[i].result);

            if ((i && i % BATCHSIZE == 0))
                update_network(opt, nn, grad, LEARNRATE, BATCHSIZE);
        }

        // update_network(opt, nn, grad, LEARNRATE, NSAMPLES);

        printf("[Epoch %5d] Loss = %.9f\n", epoch, loss / NSAMPLES);
        fflush(stdout);

        char fname[512];
        sprintf(fname, "%sepoch%d.nn", "Networks/", epoch);
        save_network(nn, fname);
    }
}

/**************************************************************************************************************/

void input_transform(Sample *sample, Matrix *matrix, Vector *bias, Vector *output) {

    for (int i = 0; i < output->length; i++)
        output->values[i] = bias->values[i];

    for (int i = 0; i < sample->length; i++)
        for (int j = 0; j < matrix->cols; j++)
            output->values[j] += matrix->values[sample->indices[i] * matrix->cols + j];
}

void affine_transform(Vector *vector, Matrix *matrix, Vector *bias, Vector *output) {

    set_vector(output, bias->values);

    for (int i = 0; i < matrix->rows; i++)
        for (int j = 0; j < matrix->cols; j++)
            output->values[j] += vector->values[i] * matrix->values[i * matrix->cols + j];
}

/**************************************************************************************************************/

Network *create_network(int length, Layer *layers) {

    Network *nn = malloc(sizeof(Network));

    nn->layers      = length;
    nn->weights     = malloc(sizeof(Matrix*) * length);
    nn->biases      = malloc(sizeof(Vector*) * length);
    nn->activations = malloc(sizeof(Activation) * length);
    nn->derivatives = malloc(sizeof(Activation) * length);

    for (int i = 0; i < length; i++) {
        nn->weights[i]     = create_matrix(layers[i].inputs, layers[i].outputs);
        nn->biases[i]      = create_vector(layers[i].outputs);
        nn->activations[i] = layers[i].activation;
        nn->derivatives[i] = layers[i].derivative;
    }

    randomize_network(nn);

    return nn;
}

void delete_network(Network *nn) {

    for (int i = 0; i < nn->layers; i++) {
        delete_matrix(nn->weights[i]);
        delete_vector(nn->biases[i]);
    }

    free(nn->weights);
    free(nn->biases);
    free(nn);
}

void print_network(Network *nn) {

    for (int i = 0; i < nn->layers; i++) {

        printf("Layer %d Weights (%dx%d)\n", i + 1, nn->weights[i]->rows, nn->weights[i]->cols);
        print_matrix(nn->weights[i]);

        printf("Layer %d Biases (%dx1)\n", i + 1, nn->biases[i]->length);
        print_vector(nn->biases[i]);
    }
}

void randomize_network(Network *nn) {

    #define random_weight() (((rand() % 10000) - 5000) / 5000.0)

    for (int i = 0; i < nn->layers; i++) {

        for (int j = 0; j < nn->weights[i]->rows * nn->weights[i]->cols; j++)
            nn->weights[i]->values[j] = random_weight();

        for (int j = 0; j < nn->biases[i]->length; j++)
            nn->biases[i]->values[j] = random_weight();
    }

    #undef random_weight
}


void save_network(Network *nn, char *fname) {

    FILE *fout = fopen(fname, "w");

    for (int layer = 0; layer < nn->layers; layer++) {

        const int rows = nn->weights[layer]->rows;
        const int cols = nn->weights[layer]->cols;

        for (int col = 0; col < cols; col++) {
            fprintf(fout, "\"%d ", rows);
            for (int row = 0; row < rows; row++)
                fprintf(fout, "%f ", nn->weights[layer]->values[row * cols + col]);
            fprintf(fout, "%f\",\n", nn->biases[layer]->values[col]);
        }
    }

    fclose(fout);
}

void load_network(Network *nn, char *fname) {
}


Evaluator *create_evaluator(Network *nn) {

    Evaluator *eval   = malloc(sizeof(Evaluator));
    eval->layers      = nn->layers;
    eval->neurons     = malloc(sizeof(Vector*) * eval->layers);
    eval->activations = malloc(sizeof(Vector*) * eval->layers);

    for (int i = 0; i < eval->layers; i++) {
        eval->neurons[i]     = create_vector(nn->biases[i]->length);
        eval->activations[i] = create_vector(nn->biases[i]->length);
    }

    return eval;
}

void delete_evaluator(Evaluator *eval) {

    for (int i = 0; i < eval->layers; i++) {
        delete_vector(eval->neurons[i]);
        delete_vector(eval->activations[i]);
    }

    free(eval->neurons);
    free(eval->activations);
    free(eval);
}

void print_evaluator(Evaluator *eval) {

    printf("==========================================\n");
    printf("======== Cached Evalution Nuerons ========\n\n");

    for (int i = 0; i < eval->layers; i++) {
        printf("Layer %d Neurons:\n", i + 1);
        print_vector(eval->neurons[i]);
        printf("Layer %d Activations:\n", i + 1);
        print_vector(eval->activations[i]);
    }

    printf("==========================================\n");
}





void sparse_evaluate_network(Network *nn, Evaluator *eval, Sample *sample) {

    assert(nn->layers == eval->layers);

    int layer = 0;

    // Input Layer

    {
        input_transform(sample, nn->weights[0], nn->biases[0], eval->neurons[0]);
        activate_layer(eval->neurons[0], eval->activations[0], nn->activations[0]);
        layer++;
    }

    // Hidden Layers

    while (layer < nn->layers - 1) {

        affine_transform(
            eval->activations[layer-1], nn->weights[layer],
            nn->biases[layer], eval->neurons[layer]
        );

        activate_layer(eval->neurons[layer], eval->activations[layer], nn->activations[layer]);
        layer++;
    }

    // Output Layer

    {
        affine_transform(
            eval->activations[layer-1], nn->weights[layer],
            nn->biases[layer], eval->neurons[layer]
        );

        activate_layer(eval->neurons[layer], eval->activations[layer], nn->activations[layer]);
    }
}

/**************************************************************************************************************/


/**************************************************************************************************************/

Gradient *create_gradient(Network *nn) {

    Gradient *grad = malloc(sizeof(Gradient));

    grad->layers  = nn->layers;
    grad->weights = malloc(sizeof(Matrix*) * grad->layers);
    grad->biases  = malloc(sizeof(Vector*) * grad->layers);

    for (int i = 0; i < grad->layers; i++) {
        grad->weights[i] = create_matrix(nn->weights[i]->rows, nn->weights[i]->cols);
        grad->biases[i]  = create_vector(nn->biases[i]->length);
    }

    return grad;
}

void delete_gradient(Gradient *grad) {
    delete_network(grad);
}

void print_gradient(Gradient *grad) {
    print_network(grad);
}

void zero_gradient(Gradient *grad) {

    for (int i = 0; i < grad->layers; i++) {

        for (int j = 0; j < grad->weights[i]->rows * grad->weights[i]->cols; j++)
            grad->weights[i]->values[j] = 0.0;

        for (int j = 0; j < grad->biases[i]->length; j++)
            grad->biases[i]->values[j] = 0.0;
    }
}


void build_backprop_grad(Network *nn, Evaluator *eval, Gradient *grad, Sample *sample) {

    const int final  = nn->layers - 1;

    float dloss_dout = loss_prime(eval->activations[final]->values[0], sample->result);

    float delta[] = { dloss_dout };

    apply_backprop(nn, eval, grad, sample, delta, final);
}

void apply_backprop(Network *nn, Evaluator *eval, Gradient *grad, Sample *sample, float *delta, int layer) {

    const int final  = nn->layers - 1;

    if (layer == 0) return apply_backprop_input(nn, eval, grad, sample, delta);

    float delta_d1[grad->weights[layer]->rows];
    float (*activation_prime)(float) = layer == final ? &sigmoid_prime : &relu_prime;

    mul_vector_func_of_vec(delta, eval->neurons[layer], activation_prime);
    add_array_to_vector(grad->biases[layer], delta);
    add_array_mul_vector_to_matrix(grad->weights[layer], delta, eval->activations[layer-1]);
    set_vector_vec_mul_mat(delta_d1, delta, nn->weights[layer]);
    apply_backprop(nn, eval, grad, sample, delta_d1, layer-1);
}

void apply_backprop_input(Network *nn, Evaluator *eval, Gradient *grad, Sample *sample, float *delta) {

    (void) nn;

    mul_vector_func_of_vec(delta, eval->neurons[0], &relu_prime);
    add_array_to_vector(grad->biases[0], delta);

    for (int i = 0; i < sample->length; i++)
        for (int j = 0; j < grad->weights[0]->cols; j++)
            grad->weights[0]->values[sample->indices[i] * grad->weights[0]->cols + j] += delta[j];
}

/**************************************************************************************************************/

Sample *load_samples(char *fname, int length) {

    Sample *samples = malloc(sizeof(Sample) * length);
    printf("Allocated %.2fMB for Samples\n",
        (float)(sizeof(Sample) * length) / (1024 * 1024));

    FILE *fin = fopen(fname, "r");

    for (int i = 0; i < length; i++) {

        load_sample(fin, &samples[i]);

        if (i == length - 1 || i % (1024*256) == 0) {
            printf("\rLoaded %d of %d Samples", i+1, length);
            fflush(stdout);
        }
    }

    printf("\nFinished Reading %s\n\n", fname);

    fclose(fin);

    return samples;
}

void load_sample(FILE *fin, Sample *sample) {

    char *ptr, line[1024];
    if (fgets(line, 1024, fin) == NULL)
        exit(EXIT_FAILURE);

    sample->length = 0;
    sample->result = atof(strtok(line, " "));

    while ((ptr = strtok(NULL, " ")) != NULL)
        sample->indices[sample->length++] = atoi(ptr);
}

/**************************************************************************************************************/

Optimizer *create_optimizer(Network *nn) {
    Optimizer *opt = malloc(sizeof(Optimizer));
    opt->momentum = create_gradient(nn);
    opt->velocity = create_gradient(nn);
    return opt;
}

void delete_optimizer(Optimizer *opt) {
    free(opt->momentum);
    free(opt->velocity);
    free(opt);
}

void update_network(Optimizer *opt, Network *nn, Gradient *grad, float lrate, int batch_size) {

    for (int layer = 0; layer < nn->layers; layer++) {

        const int rows = nn->weights[layer]->rows;
        const int cols = nn->weights[layer]->cols;

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < rows * cols; i++) {

            opt->momentum->weights[layer]->values[i]
                = (BETA_1 * opt->momentum->weights[layer]->values[i])
                + (1 - BETA_1) * (grad->weights[layer]->values[i] / batch_size);

            opt->velocity->weights[layer]->values[i]
                = (BETA_2 * opt->velocity->weights[layer]->values[i])
                + (1 - BETA_2) * pow(grad->weights[layer]->values[i] / batch_size, 2.0);

            nn->weights[layer]->values[i] -= lrate * opt->momentum->weights[layer]->values[i]
                                           * (1.0 / (1e-8 + sqrt(opt->velocity->weights[layer]->values[i])));
        }

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < nn->biases[layer]->length; i++) {

            opt->momentum->biases[layer]->values[i]
                = (BETA_1 * opt->momentum->biases[layer]->values[i])
                + (1 - BETA_1) * (grad->biases[layer]->values[i] / batch_size);

            opt->velocity->biases[layer]->values[i]
                = (BETA_2 * opt->velocity->biases[layer]->values[i])
                + (1 - BETA_2) * pow(grad->biases[layer]->values[i] / batch_size, 2.0);

            nn->biases[layer]->values[i] -= lrate * opt->momentum->biases[layer]->values[i]
                                          * (1.0 / (1e-8 + sqrt(opt->velocity->biases[layer]->values[i])));
        }
    }

    zero_gradient(grad);
}

/**************************************************************************************************************/
