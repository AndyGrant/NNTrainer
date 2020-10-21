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
    }, l2_loss_one_neuron, l2_loss_one_neuron_backprop);

    Optimizer *opt  = create_optimizer(nn);
    Evaluator *eval = create_evaluator(nn);
    Gradient *grad  = create_gradient(nn);

    for (int epoch = 0; epoch < 25000; epoch++) {

        float loss = 0.0;

        for (int i = 0; i < NSAMPLES; i++) {

            evaluate_network(nn, eval, &samples[i]);
            build_backprop_grad(nn, eval, grad, &samples[i]);
            loss += nn->loss(&samples[i], eval->activated[nn->layers-1]);

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

Network *create_network(int length, Layer *layers, Loss loss, BackProp backprop) {

    Network *nn = malloc(sizeof(Network));

    nn->layers   = length;
    nn->loss     = loss;
    nn->backprop = backprop;

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

    free(nn->weights    );
    free(nn->biases     );
    free(nn->activations);
    free(nn->derivatives);
    free(nn);
}

double random_uniform()
{
    //very important we return a number BETWEEN 0 and 1
    return (double) (rand() + 1) / (RAND_MAX + 2);
}

double random_standard_normal()
{
    //Box-Muller transform
    return sqrt(-2.0 * log(random_uniform())) * cos (2 * M_PI * random_uniform());
}

double random_normal(double mean, double sd)
{
    return (random_standard_normal() * sd) + mean;
}

void randomize_network(Network *nn) {

    /*
    We initialize the weights to have a mean of zero and a standard deviation
    equal to sqrt(2*n) where n is the number of neurons in that layer.

    Biases are initialized as zero.
    */

    for (int i = 0; i < nn->layers; i++) {

        for (int j = 0; j < nn->weights[i]->rows * nn->weights[i]->cols; j++)
            nn->weights[i]->values[j] = random_normal(0, sqrt(2 * nn->weights[i]->rows));

        for (int j = 0; j < nn->biases[i]->length; j++)
            nn->biases[i]->values[j] = 0;
    }
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

/**************************************************************************************************************/

Evaluator *create_evaluator(Network *nn) {

    Evaluator *eval   = malloc(sizeof(Evaluator));
    eval->layers      = nn->layers;
    eval->unactivated = malloc(sizeof(Vector*) * eval->layers);
    eval->activated   = malloc(sizeof(Vector*) * eval->layers);

    for (int i = 0; i < eval->layers; i++) {
        eval->unactivated[i] = create_vector(nn->biases[i]->length);
        eval->activated[i]   = create_vector(nn->biases[i]->length);
    }

    return eval;
}

void delete_evaluator(Evaluator *eval) {

    for (int i = 0; i < eval->layers; i++) {
        delete_vector(eval->unactivated[i]);
        delete_vector(eval->activated[i]);
    }

    free(eval->unactivated);
    free(eval->activated);
    free(eval);
}


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

    for (int i = 0; i < grad->layers; i++) {
        delete_matrix(grad->weights[i]);
        delete_vector(grad->biases[i]);
    }

    free(grad->weights);
    free(grad->biases );
    free(grad);
}

void zero_gradient(Gradient *grad) {

    for (int i = 0; i < grad->layers; i++) {
        zero_matrix(grad->weights[i]);
        zero_vector(grad->biases[i]);
    }
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
