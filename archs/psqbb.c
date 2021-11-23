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

#include "psqbb.h"

#include "../activate.h"
#include "../avx2.h"
#include "../batch.h"
#include "../operations.h"
#include "../trainer.h"
#include "../utils.h"

/// Definition of the Architecture

const Layer ARCHITECTURE[] = {
    {  768, 512, &activate_relu,    &backprop_relu    },
    {  512,   1, &activate_sigmoid, &backprop_sigmoid },
};

const size_t LAYER_COUNT = sizeof(ARCHITECTURE) / sizeof(Layer);

/// Any and all static helper functions for the Architecture

static int compute_input(const Sample *sample, int index, int square) {

    #define normal_encode(c, pt, sq) (64 * ((6 * (c)) + (pt)) + sq)

    int piece  = nibble_decode(index, sample->packed) % 8;
    int colour = nibble_decode(index, sample->packed) / 8;
    return normal_encode(colour, piece, square);

    #undef normal_encode
}

/// Extra helpers for the Architecture

void init_architecture(Network *nn) {
    (void) nn;
}

/// Implementation of the Architecture interface

void insert_indices(bool *array, Sample *sample) {

    uint64_t bb = sample->occupied;
    for (int i = 0; bb != 0ull; i++)
        array[compute_input(sample, i, poplsb(&bb))] = true;
}

void input_transform(const Sample *sample, const Matrix *matrix, const Vector *bias, Vector *output) {

    set_vector(output, bias->values);

    uint64_t bb = sample->occupied;

    for (int i = 0; bb != 0ull; i++) {

        int input = compute_input(sample, i, poplsb(&bb));

        for (int j = 0; j < matrix->cols; j++)
            output->values[j] += matrix->values[input * matrix->cols + j];
    }
}

void apply_backprop_input(Network *nn, Evaluator *eval, Gradient *grad, Sample *sample, float *dlossdz) {

    uint64_t bb = sample->occupied;

    nn->backprops[0](dlossdz, eval->unactivated[0], eval->activated[0]);
    add_array_to_vector(grad->biases[0], dlossdz);

    for (int i = 0; bb != 0ull; i++) {

        int index = compute_input(sample, i, poplsb(&bb));

        for (int j = 0; j < grad->weights[0]->cols; j++)
            grad->weights[0]->values[index * grad->weights[0]->cols + j] += dlossdz[j];
    }
}

void update_input_weights(Optimizer *opt, Network *nn, Gradient **grads, Batch *batch, int idx, int age) {

    const int start = batch->indices[idx] * nn->weights[0]->cols;

    for (int i = start; i < start + nn->weights[0]->cols; i += 8)
        avx2_update_weights(opt, nn, grads, 0, i, age);
}

void export_network(Network *nn) {
    (void) nn;
}