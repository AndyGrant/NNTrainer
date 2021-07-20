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

#include "halfkp.h"

#include "../activate.h"
#include "../avx2.h"
#include "../batch.h"
#include "../operations.h"
#include "../trainer.h"
#include "../utils.h"

/// Definition of the Architecture

const Layer ARCHITECTURE[] = {
    {43850, 256, &activate_relu,    &backprop_relu    },
    {  512,  32, &activate_relu,    &backprop_relu    },
    {   32,  32, &activate_relu,    &backprop_relu    },
    {   32,   1, &activate_sigmoid, &backprop_sigmoid },
};

const size_t LAYER_COUNT = sizeof(ARCHITECTURE) / sizeof(Layer);

/// Extra helpers for the Architecture

Gradient *L0Gradient;
pthread_mutex_t *L0Locks;

void init_architecture(Network *nn) {

    L0Gradient = create_gradient(nn);
    L0Locks = malloc(sizeof(pthread_mutex_t) * 40960);
    for (int i = 0; i < 40960; i++) pthread_mutex_init(&L0Locks[i], NULL);
}

/// Implementation of the Architecture interface

void insert_indices(bool *array, Sample *sample) {

    int inputs[6];
    uint64_t bb = sample->occupied;

    for (int i = 0; bb != 0ull; i++) {
        compute_inputs(sample, i, poplsb(&bb), inputs);
        for (int j = 0; j < 6; j++)
            array[inputs[j]] = true;
    }
}

void input_transform(const Sample *sample, const Matrix *matrix, const Vector *bias, Vector *output) {

    uint64_t bb = sample->occupied;
    int inputs[6], seg1_head = 0, seg2_head = matrix->cols;

    for (int i = 0; i < bias->length; i++) {
        output->values[seg1_head + i] = bias->values[i];
        output->values[seg2_head + i] = bias->values[i];
    }

    for (int i = 0; bb != 0ull; i++) {

        compute_inputs(sample, i, poplsb(&bb), inputs);

        for (int j = 0; j < matrix->cols; j++)
            output->values[seg1_head + j] += matrix->values[inputs[0] * matrix->cols + j]
                                           + matrix->values[inputs[2] * matrix->cols + j]
                                           + matrix->values[inputs[4] * matrix->cols + j];

        for (int j = 0; j < matrix->cols; j++)
            output->values[seg2_head + j] += matrix->values[inputs[1] * matrix->cols + j]
                                           + matrix->values[inputs[3] * matrix->cols + j]
                                           + matrix->values[inputs[5] * matrix->cols + j];
    }
}

void apply_backprop_input(Network *nn, Evaluator *eval, Gradient *grad, Sample *sample, float *dlossdz) {

    uint64_t bb = sample->occupied;
    int inputs[6], seg1_head = 0, seg2_head = grad->weights[0]->cols;

    __m256 *const segm1 = (__m256*) &dlossdz[seg1_head];
    __m256 *const segm2 = (__m256*) &dlossdz[seg2_head];

    nn->backprops[0](dlossdz, eval->unactivated[0], eval->activated[0]);
    for (int i = 0; i < grad->biases[0]->length; i++)
        grad->biases[0]->values[i] += dlossdz[seg1_head+i] + dlossdz[seg2_head+i];

    for (int i = 0; bb != 0ull; i++) {

        compute_inputs(sample, i, poplsb(&bb), inputs);

        __m256* grad0 = (__m256*) &L0Gradient->weights[0]->values[inputs[0] * grad->weights[0]->cols];
        __m256* grad1 = (__m256*) &L0Gradient->weights[0]->values[inputs[1] * grad->weights[0]->cols];

        __m256* grad2 = (__m256*) &grad->weights[0]->values[inputs[2] * grad->weights[0]->cols];
        __m256* grad4 = (__m256*) &grad->weights[0]->values[inputs[4] * grad->weights[0]->cols];
        __m256* grad3 = (__m256*) &grad->weights[0]->values[inputs[3] * grad->weights[0]->cols];
        __m256* grad5 = (__m256*) &grad->weights[0]->values[inputs[5] * grad->weights[0]->cols];

        pthread_mutex_lock(&L0Locks[inputs[0]]);
        for (int j = 0; j < grad->weights[0]->cols / 8; j+=4) {
            grad0[j+0] = _mm256_add_ps(grad0[j+0], segm1[j+0]);
            grad0[j+1] = _mm256_add_ps(grad0[j+1], segm1[j+1]);
            grad0[j+2] = _mm256_add_ps(grad0[j+2], segm1[j+2]);
            grad0[j+3] = _mm256_add_ps(grad0[j+3], segm1[j+3]);
        } pthread_mutex_unlock(&L0Locks[inputs[0]]);

        pthread_mutex_lock(&L0Locks[inputs[1]]);
        for (int j = 0; j < grad->weights[0]->cols / 8; j+=4) {
            grad1[j+0] = _mm256_add_ps(grad1[j+0], segm2[j+0]);
            grad1[j+1] = _mm256_add_ps(grad1[j+1], segm2[j+1]);
            grad1[j+2] = _mm256_add_ps(grad1[j+2], segm2[j+2]);
            grad1[j+3] = _mm256_add_ps(grad1[j+3], segm2[j+3]);
        } pthread_mutex_unlock(&L0Locks[inputs[1]]);

        for (int j = 0; j < grad->weights[0]->cols / 8; j++) {
            grad2[j] = _mm256_add_ps(grad2[j], segm1[j]);
            grad4[j] = _mm256_add_ps(grad4[j], segm1[j]);
            grad3[j] = _mm256_add_ps(grad3[j], segm2[j]);
            grad5[j] = _mm256_add_ps(grad5[j], segm2[j]);
        }
    }
}

void update_input_weights(Optimizer *opt, Network *nn, Gradient **grads, Batch *batch, int idx, int age) {

    const int start = batch->indices[idx] * nn->weights[0]->cols;

    if (batch->indices[idx] >= 40960)
        for (int i = start; i < start + nn->weights[0]->cols; i += 8)
            avx2_update_weights(opt, nn, grads, 0, i, age);

    if (batch->indices[idx] < 40960)
        for (int i = start; i < start + nn->weights[0]->cols; i += 64)
            avx2_update_8x8(opt, nn, &L0Gradient, 0, i, age);
}