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
#include <stdint.h>
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
#include "trainer.h"
#include "vector.h"

#define USE_AVX2 1

int NTHREADS;

#if NN_TYPE == HALFKP

void collapse_network(Network *nn) {

    const int rows = nn->weights[0]->rows = 40960;
    const int cols = nn->weights[0]->cols;

    for (int i = 0; i < rows; i++) {

        const int offset  = cols * i;
        const int augoff1 = cols * (rows + 0    + nnue_to_relative_kmap(i));
        const int augoff2 = cols * (rows + 2250 + nnue_to_relative_psqt(i));

        for (int j = 0; j < cols; j++)
            nn->weights[0]->values[offset+j] += nn->weights[0]->values[augoff1+j]
                                              + nn->weights[0]->values[augoff2+j];
    }
}

void export_network(Network *nn) {

    FILE *fout = fopen("exported.nn", "wb");

    collapse_network(nn); // Collapse the Factorizer Inputs

    {
        #define CLAMP1KB(f)   ((f) > 2048 ? 2048 : ((f) < -2048 ? -2048 : (f)))
        #define QUANTIN16B(f) ((int16_t) CLAMP1KB(roundf(64.0 * (f))))
        #define QUANTIN16W(f) ((int16_t) CLAMP1KB(roundf(64.0 * (f))))

        const int layer = 0;
        const int rows  = nn->weights[layer]->rows;
        const int cols  = nn->weights[layer]->cols;

        int16_t *biases  = malloc(sizeof(int16_t) * cols);
        int16_t *weights = malloc(sizeof(int16_t) * rows * cols);

        for (int i = 0; i < cols; i++)
            biases[i] = QUANTIN16B(nn->biases[layer]->values[i]);

        for (int i = 0; i < rows * cols; i++)
            weights[i] = QUANTIN16W(nn->weights[layer]->values[i]);

        fwrite(biases, sizeof(int16_t), cols, fout);
        fwrite(weights, sizeof(int16_t), rows * cols, fout);
        free(biases); free(weights);
    }

    {
        #define QUANT32B(f) ((int16_t) (roundf(64.0 * (f))))
        #define QUANT16W(f) ((int32_t) (roundf(64.0 * (f))))

        const int layer = 1;
        const int rows  = nn->weights[layer]->rows;
        const int cols  = nn->weights[layer]->cols;

        int32_t *biases  = malloc(sizeof(int32_t) * cols);
        int16_t *weights = malloc(sizeof(int16_t) * rows * cols);

        for (int i = 0; i < cols; i++)
            biases[i] = QUANT32B(nn->biases[layer]->values[i]);

        for (int i = 0; i < rows * cols; i++)
            weights[i] = QUANT16W(nn->weights[layer]->values[i]);

        fwrite(biases, sizeof(int32_t), cols, fout);
        fwrite(weights, sizeof(int16_t), rows * cols, fout);
        free(biases); free(weights);
    }

    for (int layer = 2; layer < nn->layers; layer++) {

        const int rows = nn->weights[layer]->rows;
        const int cols = nn->weights[layer]->cols;

        fwrite(nn->biases[layer]->values, sizeof(float), cols, fout);
        fwrite(nn->weights[layer]->values, sizeof(float), rows * cols, fout);
    }

    fclose(fout);
}

#endif

int main() {

    // const size_t length = sizeof(ARCHITECTURE) / sizeof(Layer);
    // Network *nn = create_network(length, ARCHITECTURE);
    // load_network(nn, "x128.nn");
    // export_network(nn);
    // return 1;

    setvbuf(stdout, NULL, _IONBF, 0);
    NTHREADS = omp_get_max_threads();
    printf("Found %d Threads to train with\n", NTHREADS);

    const size_t length = sizeof(ARCHITECTURE) / sizeof(Layer);
    Network *nn = create_network(length, ARCHITECTURE);
    if (USE_WEIGHTS) load_network(nn, NNWEIGHTS);
    else printf("Created Network with randomized Weights\n\n");

    printf("Loading Validation Dataset...\n");
    Sample *validate = load_samples(VALIDFILE, NVALIDATE);

    printf("Loading Training Dataset...\n");
    Sample *samples  = load_samples(DATAFILE, NSAMPLES);

    Batch *batches = create_batches(samples, NSAMPLES, BATCHSIZE);
    Optimizer *opt = create_optimizer(nn);

    Evaluator *evals[NTHREADS];
    Gradient  *grads[NTHREADS];

    for (int i = 0; i < NTHREADS; i++)
        evals[i] = create_evaluator(nn);

    for (int i = 0; i < NTHREADS; i++)
        grads[i] = create_gradient(nn);

    for (int epoch = 0; epoch < 25000; epoch++) {

        double loss = 0.0, vloss = 0.0;
        double start = get_time_point();

        /// Train by iterating over each of the Training Samples

        for (int batch = 0; batch < NSAMPLES / BATCHSIZE; batch++) {

            #pragma omp parallel for schedule(static, BATCHSIZE / NTHREADS) num_threads(NTHREADS) reduction(+:loss)
            for (int i = batch * BATCHSIZE; i < (batch+1) * BATCHSIZE; i++) {
                const int tidx = omp_get_thread_num();
                evaluate_network(nn, evals[tidx], &samples[i]);
                build_backprop_grad(nn, evals[tidx], grads[tidx], &samples[i]);
                loss += LOSS_FUNC(&samples[i], evals[tidx]->activated[nn->layers-1]);
            }

            update_network(opt, nn, grads, &batches[batch]);

            if (batch % 64 == 0) {
                double elapsed = (get_time_point() - start) / 1000.0;
                printf("\r[%4d] [%8.3fs] [Batch %d / %d]", epoch, elapsed, batch, NSAMPLES / BATCHSIZE);
            }
        }

        double elapsed = (get_time_point() - start) / 1000.0;

        /// Verify by iterating over each of the Validation Samples

        #pragma omp parallel for schedule(static, NVALIDATE / NTHREADS) num_threads(NTHREADS) reduction(+:vloss)
        for (int i = 0; i < NVALIDATE; i++) {
            const int tidx = omp_get_thread_num();
            evaluate_network(nn, evals[tidx], &validate[i]);
            vloss += LOSS_FUNC(&validate[i], evals[tidx]->activated[nn->layers-1]);
        }

        printf("\r[%4d] [%8.3fs] [Training = %2.10f] [Validation = %2.10f]\n",
            epoch, elapsed, loss / NSAMPLES, vloss / NVALIDATE);

        char fname[512];
        sprintf(fname, "%sepoch%d.nn", "Networks/", epoch);
        save_network(nn, fname);
    }
}

/**************************************************************************************************************/

Network *create_network(int length, const Layer *layers) {

    Network *nn     = malloc(sizeof(Network));
    nn->weights     = malloc(sizeof(Matrix*   ) * length);
    nn->biases      = malloc(sizeof(Vector*   ) * length);
    nn->activations = malloc(sizeof(Activation) * length);
    nn->backprops   = malloc(sizeof(BackProp  ) * length);
    nn->layers      = length;

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
            nn->weights[i]->values[j] = random() / 4.0;

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

    printf("Created Network with Weights from %s\n\n", fname);

    fclose(fin);
}

/**************************************************************************************************************/

Sample *load_samples(const char *fname, int length) {

    Sample *samples = calloc(length, sizeof(Sample));
    printf("Allocated %.2fMB for Samples\n",
        (float)(sizeof(Sample) * length) / (1024 * 1024));

    FILE *fin = fopen(fname, "rb");

    for (int i = 0; i < length; i++) {
        load_sample(fin, &samples[i]);
        if (i == length - 1 || i % (1024*256) == 0)
            printf("\rLoaded %d of %d Samples", i+1, length);
    }

    printf("\nFinished Reading %s\n\n", fname);

    fclose(fin);

    return samples;
}

void load_sample(FILE *fin, Sample *sample) {

    int16_t eval;
    uint64_t pieces;
    uint8_t result, turn, N, wksq, bksq, packed[16];

    fread(&pieces, sizeof(uint64_t), 1, fin);
    fread(&eval,   sizeof(int16_t ), 1, fin);
    fread(&result, sizeof(uint8_t ), 1, fin);
    fread(&turn,   sizeof(uint8_t ), 1, fin);
    fread(&wksq,   sizeof(uint8_t ), 1, fin);
    fread(&bksq,   sizeof(uint8_t ), 1, fin);
    fread(&N,      sizeof(uint8_t ), 1, fin);
    fread(packed,  sizeof(uint8_t ), (N + 1) / 2, fin);

    #define nibble_decode(i, A) (((i) % 2) ? (A[(i)/2] & 0xF) : (A[(i)/2]) >> 4)
    #define nibble_encode(i, A, cp) (A[(i)/2] |= (((i) % 2) ? (cp) : (cp << 4)))
    #define normal_encode(c, pt, sq) (64 * ((6 * (c)) + (pt)) + sq)

#if NN_TYPE == NORMAL

    sample->label  = sigmoid(eval);
    sample->length = 0;

    for (int i = 0; pieces; i++) {

        uint8_t enc = nibble_decode(i, packed);

        int sq = poplsb(&pieces);
        int c  = enc / 8, pt = enc % 8;

        sample->indices[sample->length++] = normal_encode(c, pt, sq);
    }

#elif NN_TYPE == HALFKP

    sample->occupied = pieces;
    sample->turn     = turn;
    sample->wking    = wksq;
    sample->bking    = bksq;

    for (int i = 0, j = 0; pieces != 0ull; i++, j++) {

        uint8_t cpdata = nibble_decode(i, packed);
        int sq = poplsb(&pieces), pt = cpdata % 8;

        if (pt != KING) nibble_encode(j, sample->packed, cpdata);
        if (pt == KING) { sample->occupied ^= 1ull << sq; j--; }
    }

    sample->eval   = sample->turn ? -eval : eval;
    sample->result = sample->turn ? 2 - result : result;

#endif

    #undef nibble_decode
    #undef nibble_encode
    #undef normal_encode
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
