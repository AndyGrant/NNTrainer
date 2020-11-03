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
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "avx2.h"
#include "activate.h"
#include "batch.h"
#include "config.h"
#include "evaluator.h"
#include "gradient.h"
#include "matrix.h"
#include "operations.h"
#include "optimizer.h"
#include "timing.h"
#include "trainer.h"
#include "vector.h"

#define USE_AVX2 1

int NTHREADS;

int main() {

    NTHREADS = omp_get_max_threads();
    printf("Found %d Threads to Train on\n\n", NTHREADS);

    const size_t length = sizeof(ARCHITECTURE) / sizeof(Layer);
    Network *nn = create_network(length, ARCHITECTURE, LOSS_FUNC, LOSSPROP_FUNC, NN_TYPE);
    if (USE_WEIGHTS) load_network(nn, NNWEIGHTS);

    Sample *samples = load_samples(DATAFILE, NSAMPLES);
    Batch  *batches = create_batches(samples, NSAMPLES, BATCHSIZE);

    Optimizer *opt  = create_optimizer(nn);
    Evaluator *evals[NTHREADS];
    Gradient  *grads[NTHREADS];

    for (int i = 0; i < NTHREADS; i++)
        evals[i] = create_evaluator(nn);

    for (int i = 0; i < NTHREADS; i++)
        grads[i] = create_gradient(nn);

    for (int epoch = 0; epoch < 25000; epoch++) {

        float loss = 0.0;

        double start = get_time_point();

        for (int batch = 0; batch < NSAMPLES / BATCHSIZE; batch++) {

            #pragma omp parallel for schedule(static, BATCHSIZE / NTHREADS) num_threads(NTHREADS) reduction(+:loss)
            for (int i = batch * BATCHSIZE; i < (batch+1) * BATCHSIZE; i++) {
                const int tidx = omp_get_thread_num();
                evaluate_network(nn, evals[tidx], &samples[i]);
                build_backprop_grad(nn, evals[tidx], grads[tidx], &samples[i]);
                loss += nn->loss(&samples[i], evals[tidx]->activated[nn->layers-1]);
            }

            update_network(opt, nn, grads, &batches[batch]);
        }

        double elapsed = (get_time_point() - start) / 1000.0;

        printf("[%4d] [%0.3fs] Loss = %.9f\n", epoch, elapsed, loss / NSAMPLES);
        fflush(stdout);

        char fname[512];
        sprintf(fname, "%sepoch%d.nn", "Networks/", epoch);
        save_network(nn, fname);
    }
}

/**************************************************************************************************************/

Network *create_network(int length, const Layer *layers, Loss loss, LossProp lossprop, int type) {

    Network *nn = malloc(sizeof(Network));

    nn->layers   = length;
    nn->loss     = loss;
    nn->lossprop = lossprop;
    nn->type     = type;

    nn->weights     = malloc(sizeof(Matrix*   ) * length);
    nn->biases      = malloc(sizeof(Vector*   ) * length);
    nn->activations = malloc(sizeof(Activation) * length);
    nn->backprops   = malloc(sizeof(BackProp  ) * length);

    for (int i = 0; i < length; i++) {
        nn->weights[i]     = create_matrix(layers[i].inputs, layers[i].outputs);
        nn->biases[i]      = create_vector(layers[i].outputs);
        nn->activations[i] = layers[i].activation;
        nn->backprops[i]   = layers[i].backprop;
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
    free(nn->backprops  );
    free(nn);
}

void randomize_network(Network *nn) {

    #define uniform() ((float) (rand() + 1) / ((float) RAND_MAX + 2))
    #define random()  (sqrtf(-2.0 * log(uniform())) * cos(2 * M_PI * uniform()))

    for (int i = 0; i < nn->layers; i++)
        for (int j = 0; j < nn->weights[i]->rows * nn->weights[i]->cols; j++)
            nn->weights[i]->values[j] = random();

    #undef uniform
    #undef random
}

void save_network(Network *nn, const char *fname) {

    FILE *fout = fopen(fname, "wb");

    for (int layer = 0; layer < nn->layers; layer++) {

        const int rows = nn->weights[layer]->rows;
        const int cols = nn->weights[layer]->cols;

        fwrite(nn->biases[layer]->values, sizeof(float), cols, fout);
        fwrite(nn->weights[layer]->values, sizeof(float), rows * cols, fout);
    }

    fclose(fout);
}

void load_network(Network *nn, const char *fname) {

    FILE *fin = fopen(fname, "rb");

    for (int layer = 0; layer < nn->layers; layer++) {

        const int rows = nn->weights[layer]->rows;
        const int cols = nn->weights[layer]->cols;

        if (   fread(nn->biases[layer]->values, sizeof(float), cols, fin) != (size_t) cols
            || fread(nn->weights[layer]->values, sizeof(float), rows * cols, fin) != (size_t) rows * cols)
            exit(EXIT_FAILURE);
    }

    printf("Loaded Weights from %s\n", fname); fflush(stdout);

    fclose(fin);
}

/**************************************************************************************************************/

Sample *load_samples(const char *fname, int length) {

    Sample *samples = malloc(sizeof(Sample) * length);
    printf("Allocated %.2fMB for Samples\n",
        (float)(sizeof(Sample) * length) / (1024 * 1024));
    fflush(stdout);

    FILE *fin = fopen(fname, "r");

    for (int i = 0; i < length; i++) {

        load_sample(fin, &samples[i]);

        if (i == length - 1 || i % (1024*256) == 0) {
            printf("\rLoaded %d of %d Samples", i+1, length);
            fflush(stdout);
        }
    }

    printf("\nFinished Reading %s\n\n", fname);
    fflush(stdout);

    fclose(fin);

    return samples;
}

void load_sample(FILE *fin, Sample *sample) {

    char *ptr, line[1024];
    if (fgets(line, 1024, fin) == NULL)
        exit(EXIT_FAILURE);

    sample->label = atof(strtok(line, " "));
    sample->turn  = atoi(strtok(NULL, " "));
    sample->wking = atoi(strtok(NULL, " "));
    sample->bking = atoi(strtok(NULL, " "));

    sample->length = 0;
    while ((ptr = strtok(NULL, " ")) != NULL)
        sample->indices[sample->length++] = atoi(ptr);

    if (sample->turn) sample->label = -sample->label;
}

/**************************************************************************************************************/

float accumulate_grad_weight(Gradient **grads, int layer, int idx) {

    float total = 0.0;

    for (int i = 0; i < NTHREADS; i++)
        total += grads[i]->weights[layer]->values[idx];

    return total;
}

float accumulate_grad_bias(Gradient **grads, int layer, int idx) {

    float total = 0.0;

    for (int i = 0; i < NTHREADS; i++)
        total += grads[i]->biases[layer]->values[idx];

    return total;
}

void update_network(Optimizer *opt, Network *nn, Gradient **grads, Batch *batch) {

    #pragma omp parallel for schedule(static) num_threads(NTHREADS)
    for (int idx = 0; idx < batch->inputs; idx++) {

        int start = batch->indices[idx] * nn->weights[0]->cols;

        if (USE_AVX2 && nn->weights[0]->cols % 8 == 0) {
            for (int i = start; i < start + nn->weights[0]->cols; i += 8)
                avx2_update_weights(opt, nn, grads, 0, i);
        }

        else {

            for (int i = start; i < start + nn->weights[0]->cols; i++) {

                const float true_grad = accumulate_grad_weight(grads, 0, i) / BATCHSIZE;

                opt->momentum->weights[0]->values[i]
                    = (BETA_1 * opt->momentum->weights[0]->values[i])
                    + (1 - BETA_1) * true_grad;

                opt->velocity->weights[0]->values[i]
                    = (BETA_2 * opt->velocity->weights[0]->values[i])
                    + (1 - BETA_2) * pow(true_grad, 2.0);

                nn->weights[0]->values[i] -= LEARNRATE * opt->momentum->weights[0]->values[i]
                                           * (1.0 / (1e-8 + sqrt(opt->velocity->weights[0]->values[i])));
            }
        }
    }

    if (!USE_AVX2 || nn->weights[0]->cols % 8 != 0) {
        #pragma omp parallel for schedule(static) num_threads(NTHREADS)
        for (int i = 0; i < NTHREADS; i++)
            zero_matrix(grads[i]->weights[0]);
    }

    for (int layer = 1; layer < nn->layers; layer++) {

        const int rows = nn->weights[layer]->rows;
        const int cols = nn->weights[layer]->cols;

        #pragma omp parallel for schedule(static) num_threads(NTHREADS)
        for (int i = 0; i < rows * cols; i++) {

            const float true_grad = accumulate_grad_weight(grads, layer, i) / BATCHSIZE;

            opt->momentum->weights[layer]->values[i]
                = (BETA_1 * opt->momentum->weights[layer]->values[i])
                + (1 - BETA_1) * true_grad;

            opt->velocity->weights[layer]->values[i]
                = (BETA_2 * opt->velocity->weights[layer]->values[i])
                + (1 - BETA_2) * powf(true_grad, 2.0);

            nn->weights[layer]->values[i] -= LEARNRATE * opt->momentum->weights[layer]->values[i]
                                           * (1.0 / (1e-8 + sqrtf(opt->velocity->weights[layer]->values[i])));
        }

        #pragma omp parallel for schedule(static) num_threads(NTHREADS)
        for (int i = 0; i < NTHREADS; i++)
            zero_matrix(grads[i]->weights[layer]);
    }

    for (int layer = 0; layer < nn->layers; layer++) {

        #pragma omp parallel for schedule(static) num_threads(NTHREADS)
        for (int i = 0; i < nn->biases[layer]->length; i++) {

            const float true_grad = accumulate_grad_bias(grads, layer, i) / BATCHSIZE;

            opt->momentum->biases[layer]->values[i]
                = (BETA_1 * opt->momentum->biases[layer]->values[i])
                + (1 - BETA_1) * true_grad;

            opt->velocity->biases[layer]->values[i]
                = (BETA_2 * opt->velocity->biases[layer]->values[i])
                + (1 - BETA_2) * powf(true_grad, 2.0);

            nn->biases[layer]->values[i] -= LEARNRATE * opt->momentum->biases[layer]->values[i]
                                          * (1.0 / (1e-8 + sqrtf(opt->velocity->biases[layer]->values[i])));
        }

        #pragma omp parallel for schedule(static) num_threads(NTHREADS)
        for (int i = 0; i < NTHREADS; i++)
            zero_vector(grads[i]->biases[layer]);
    }
}

/**************************************************************************************************************/
