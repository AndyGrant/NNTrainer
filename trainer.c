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
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if NN_TYPE == HALFKP
    #include "archs/mirrorhkp.h"
#endif

#include "avx2.h"
#include "activate.h"
#include "batch.h"
#include "config.h"
#include "operations.h"
#include "trainer.h"
#include "utils.h"

extern const Layer ARCHITECTURE[];
extern const size_t LAYER_COUNT;

int NTHREADS;

int main(int argc, char **argv) {

    if (argc > 2 && !strcmp(argv[1], "export")) {
        Network *nn = create_network(LAYER_COUNT, ARCHITECTURE);
        export_network(nn, argv[2]);
        exit(EXIT_SUCCESS);
    }

    if (argc > 2 && !strcmp(argv[1], "import")) {
        Network *nn = create_network(LAYER_COUNT, ARCHITECTURE);
        import_network(nn, argv[2]);
        save_network(nn, "imported.nn");
        exit(EXIT_SUCCESS);
    }

    setvbuf(stdout, NULL, _IONBF, 0);
    NTHREADS = omp_get_max_threads() / 2;
    printf("Using %d Threads\n", NTHREADS);

    Network *nn = create_network(LAYER_COUNT, ARCHITECTURE);
    if (USE_WEIGHTS) load_network(nn, NNWEIGHTS);
    else printf("Created Network with randomized Weights\n\n");

    Sample *samples  = malloc(sizeof(Sample) * NSAMPLES);

    Optimizer *opt = create_optimizer(nn);
    Evaluator *evals[NTHREADS]; Gradient *grads[NTHREADS];
    for (int i = 0; i < NTHREADS; i++) evals[i] = create_evaluator(nn);
    for (int i = 0; i < NTHREADS; i++) grads[i] = create_gradient(nn);

    init_architecture(nn); // Call any Architecture Specific Inits
    if (USE_STATE) load_optimizer(opt, NNSTATE);

    for (int epoch = START_EPOCH; epoch < 25000; epoch++) {

        double loss = 0.0;
        double start = get_time_point();

        get_next_samples(DATAFILE, samples, NSAMPLES, epoch);
        Batch *batches = create_batches(samples, NSAMPLES, BATCHSIZE);

        for (int batch = 0; batch < (int)(NSAMPLES / BATCHSIZE); batch++) {

            opt->iteration++;

            #pragma omp parallel for schedule(static) num_threads(NTHREADS) reduction(+:loss)
            for (int i = batch * BATCHSIZE; i < (batch+1) * BATCHSIZE; i++) {
                const int tidx = omp_get_thread_num();
                evaluate_network(nn, evals[tidx], &samples[i]);
                build_backprop_grad(nn, evals[tidx], grads[tidx], &samples[i]);
                loss += LOSS_FUNC(&samples[i], evals[tidx]->activated[nn->layers-1]);
            }

            update_network(opt, nn, grads, &batches[batch]);

            if (batch % 64 == 0) {
                double elapsed = (get_time_point() - start) / 1000.0;
                printf("\r[%4d] [%8.2fs] [Batch %d / %d]",
                    epoch, elapsed, batch, (int) (NSAMPLES / BATCHSIZE));
            }
        }

        delete_batches(batches, NSAMPLES, BATCHSIZE);

        double elapsed = (get_time_point() - start) / 1000.0;

        printf("\r[%4d] [%8.2fs] Training [ %2.8f ]\n", epoch, elapsed, loss / NSAMPLES);

        char fname[512];

        // Save the Network in an uncollapsed, non-Quantized way
        sprintf(fname, "%sepoch%d.nn", "Networks/", epoch);
        save_network(nn, fname);

        // Save the Optimizer moments, and "Lazy" Adam trackers
        sprintf(fname, "%sepoch%d.state", "Networks/", epoch);
        save_optimizer(opt, fname);
    }
}

/**************************************************************************************************************/

Network *create_network(int length, const Layer *layers) {

    Network *nn     = malloc(sizeof(Network));
    nn->biases      = malloc(sizeof(Vector*   ) * length);
    nn->weights     = malloc(sizeof(Matrix*   ) * length);
    nn->weights_t   = malloc(sizeof(Matrix*   ) * length);
    nn->activations = malloc(sizeof(Activation) * length);
    nn->backprops   = malloc(sizeof(BackProp  ) * length);
    nn->layers      = length;

    for (int i = 0; i < length; i++) {

        nn->biases[i]      = create_vector(layers[i].outputs);
        nn->weights[i]     = create_matrix(layers[i].inputs, layers[i].outputs);
        nn->weights_t[i]   = create_matrix(layers[i].inputs, layers[i].outputs);

        nn->activations[i] = layers[i].activation;
        nn->backprops[i]   = layers[i].backprop;
    }

    randomize_network(nn);
    update_network_transposed(nn);

    return nn;
}

void update_network_transposed(Network *nn) {

    collapse_input_layer(nn); // Might be (void)

    for (int layer = 1; layer < nn->layers; layer++) {

        const int rows = nn->weights[layer]->rows;
        const int cols = nn->weights[layer]->cols;

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                nn->weights_t[layer]->values[j * rows + i] = nn->weights[layer]->values[i * cols + j];
    }
}


void delete_network(Network *nn) {

    for (int i = 0; i < nn->layers; i++) {
        delete_vector(nn->biases[i]);
        delete_matrix(nn->weights[i]);
        delete_matrix(nn->weights_t[i]);
    }

    free(nn->biases     );
    free(nn->weights    );
    free(nn->weights_t  );
    free(nn->activations);
    free(nn->backprops  );
    free(nn             );
}

void randomize_network(Network *nn) {

    #define uniform()  ((double) (rand() + 1) / ((double) RAND_MAX + 2))
    #define random()   (sqrt(-2.0 * log(uniform())) * cos(2 * M_PI * uniform()))
    #define kaiming(L) ((double)((L) ? nn->weights[L]->rows : 96.0))

    srand(time(NULL));

    for (int i = 0; i < nn->layers; i++)
        for (int j = 0; j < nn->weights[i]->rows * nn->weights[i]->cols; j++)
            nn->weights[i]->values[j] = random() * sqrt(2.0 / kaiming(i));

    #undef uniform
    #undef random
    #undef kaiming
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

    update_network_transposed(nn); // Init for nn->weights_t

    printf("Created Network with Weights from %s\n\n", fname);

    fclose(fin);
}

/**************************************************************************************************************/

Sample *get_next_samples(const char *format, Sample *samples, int length, int epoch) {

    char fname[256];
    sprintf(fname, format, epoch % NDATAFILES);
    return load_samples(fname, samples, length);
}

Sample *load_samples(const char *fname, Sample *samples, int length) {

    printf("Loading from %s\n", fname);

    FILE *fin = fopen(fname, "rb");

    if (samples == NULL)
        samples = malloc(sizeof(Sample) * length);

    if (fread(samples, sizeof(Sample), length, fin) != (size_t) length)
        exit(EXIT_FAILURE);

    fclose(fin);

    return samples;
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
        const int age = opt->iteration - opt->last_seen[batch->indices[idx]];
        update_input_weights(opt, nn, grads, batch, idx, age);
        opt->last_seen[batch->indices[idx]] = opt->iteration;
    }

    for (int layer = 1; layer < nn->layers; layer++) {

        const int rows = nn->weights[layer]->rows;
        const int cols = nn->weights[layer]->cols;

        #pragma omp parallel for schedule(static) num_threads(NTHREADS)
        for (int i = 0; i < rows * cols; i++) {

            const float true_grad = accumulate_grad_weight(grads, layer, i);

            opt->momentum->weights[layer]->values[i]
                = (BETA_1 * opt->momentum->weights[layer]->values[i])
                + ((1 - BETA_1) * LEARNRATE) * true_grad;

            opt->velocity->weights[layer]->values[i]
                = (BETA_2 * opt->velocity->weights[layer]->values[i])
                + (1 - BETA_2) * powf(true_grad, 2.0);

            nn->weights[layer]->values[i] -= opt->momentum->weights[layer]->values[i]
                                           * (1.0 / (1e-8 + sqrtf(opt->velocity->weights[layer]->values[i])));
        }

        #pragma omp parallel for schedule(static) num_threads(NTHREADS)
        for (int i = 0; i < NTHREADS; i++)
            zero_matrix(grads[i]->weights[layer]);
    }

    for (int layer = 0; layer < nn->layers; layer++) {

        #pragma omp parallel for schedule(static) num_threads(NTHREADS)
        for (int i = 0; i < nn->biases[layer]->length; i++) {

            const float true_grad = accumulate_grad_bias(grads, layer, i);

            opt->momentum->biases[layer]->values[i]
                = (BETA_1 * opt->momentum->biases[layer]->values[i])
                + ((1 - BETA_1) * LEARNRATE) * true_grad;

            opt->velocity->biases[layer]->values[i]
                = (BETA_2 * opt->velocity->biases[layer]->values[i])
                + (1 - BETA_2) * powf(true_grad, 2.0);

            nn->biases[layer]->values[i] -= opt->momentum->biases[layer]->values[i]
                                          * (1.0 / (1e-8 + sqrtf(opt->velocity->biases[layer]->values[i])));
        }

        #pragma omp parallel for schedule(static) num_threads(NTHREADS)
        for (int i = 0; i < NTHREADS; i++)
            zero_vector(grads[i]->biases[layer]);
    }

    /// Clip Layer 1's Weights ( which will be int8_t )
    /// This code should be in archs/mirrorhkp.x as a post-update supplement

    for (int i = 0; i < nn->weights[1]->rows * nn->weights[1]->cols; i++) {
        nn->weights[1]->values[i] = MIN(+3.96, nn->weights[1]->values[i]);
        nn->weights[1]->values[i] = MAX(-3.96, nn->weights[1]->values[i]);
    }

    update_network_transposed(nn); // Init for nn->weights_t
}

/**************************************************************************************************************/
