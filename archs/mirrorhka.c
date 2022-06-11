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

#include "mirrorhka.h"

#include "../activate.h"
#include "../avx2.h"
#include "../batch.h"
#include "../operations.h"
#include "../trainer.h"
#include "../utils.h"

extern int NTHREADS;

/// Definition of the Architecture

const Layer ARCHITECTURE[] = {
    {28044, 256, &activate_relu,    &backprop_relu    },
    {  512,  16, &activate_relu,    &backprop_relu    },
    {   16,  16, &activate_relu,    &backprop_relu    },
    {   16,   1, &activate_sigmoid, &backprop_sigmoid },
};

const size_t LAYER_COUNT = sizeof(ARCHITECTURE) / sizeof(Layer);

/// Any and all static helper functions for the Architecture

static void compute_inputs(const Sample *sample, int index, int square, int *inputs) {

    int stmk  = sample->turn == WHITE ? sample->wking : sample->bking;
    int nstmk = sample->turn == WHITE ? sample->bking : sample->wking;

    int sksq  = relative_square( sample->turn,  stmk);
    int nsksq = relative_square(!sample->turn, nstmk);

    int srelsq  = relative_square( sample->turn, square);
    int nsrelsq = relative_square(!sample->turn, square);

    if (queen_side_sq(sksq)) { sksq = mirror_square(sksq); srelsq = mirror_square(srelsq); }
    if (queen_side_sq(nsksq)) { nsksq = mirror_square(nsksq); nsrelsq = mirror_square(nsrelsq); }

    int piece  = nibble_decode(index, sample->packed) % 8;
    int colour = nibble_decode(index, sample->packed) / 8;

    int saug  = 15 * (7 + rank_of( sksq) - rank_of( srelsq)) + (7 + file_of( sksq) - file_of( srelsq));
    int nsaug = 15 * (7 + rank_of(nsksq) - rank_of(nsrelsq)) + (7 + file_of(nsksq) - file_of(nsrelsq));

    inputs[0] = (64 * 12 * sq64_to_sq32(sksq )) + (64 * (6 * (colour == sample->turn) + piece)) + srelsq;
    inputs[1] = (64 * 12 * sq64_to_sq32(nsksq)) + (64 * (6 * (colour != sample->turn) + piece)) + nsrelsq;

    inputs[2] = (64 * 12 * 32) + (225 * (6 * (colour == sample->turn) + piece)) + saug;
    inputs[3] = (64 * 12 * 32) + (225 * (6 * (colour != sample->turn) + piece)) + nsaug;

    inputs[4] = (64 * 12 * 32) + (225 * 12) + 64 * (6 * (colour == sample->turn) + piece) + srelsq;
    inputs[5] = (64 * 12 * 32) + (225 * 12) + 64 * (6 * (colour != sample->turn) + piece) + nsrelsq;

    assert(    0 <= inputs[0] && inputs[0] < 24576);
    assert(    0 <= inputs[1] && inputs[1] < 24576);
    assert(24576 <= inputs[2] && inputs[2] < 27276);
    assert(24576 <= inputs[3] && inputs[3] < 27276);
    assert(27276 <= inputs[4] && inputs[4] < 28044);
    assert(27276 <= inputs[5] && inputs[5] < 28044);
}

static void compute_real_inputs(const Sample *sample, int index, int square, int *inputs) {

    int stmk  = sample->turn == WHITE ? sample->wking : sample->bking;
    int nstmk = sample->turn == WHITE ? sample->bking : sample->wking;

    int sksq  = relative_square( sample->turn,  stmk);
    int nsksq = relative_square(!sample->turn, nstmk);

    int srelsq  = relative_square( sample->turn, square);
    int nsrelsq = relative_square(!sample->turn, square);

    if (queen_side_sq(sksq)) { sksq = mirror_square(sksq); srelsq = mirror_square(srelsq); }
    if (queen_side_sq(nsksq)) { nsksq = mirror_square(nsksq); nsrelsq = mirror_square(nsrelsq); }

    int piece  = nibble_decode(index, sample->packed) % 8;
    int colour = nibble_decode(index, sample->packed) / 8;

    inputs[0] = (64 * 12 * sq64_to_sq32(sksq )) + (64 * (6 * (colour == sample->turn) + piece)) + srelsq;
    inputs[1] = (64 * 12 * sq64_to_sq32(nsksq)) + (64 * (6 * (colour != sample->turn) + piece)) + nsrelsq;
}

static int nnue_to_relative_kmap(int encoded) {

    /// Given a value [0, 28044], which encodes a (King Sq, Piece-Col, Piece Sq),
    /// compute the relative index mapping of [0, 2700] which is the encoded form
    /// of (Piece-Col, Rankwise-distance, Filewise-distance).

    const int piecesq   = (encoded % 64);       // Enc = ( 1 * Piece Square )
    const int piececol  = (encoded % 768) / 64; //     + ( 64 * Piece-Col   )
    const int kingsq    = (encoded / 768);      //     + ( 768 * King Sq    )

    const int relative  = 15 * (7 + rank_of(sq32_to_sq64(kingsq)) - rank_of(piecesq))
                             + (7 + file_of(sq32_to_sq64(kingsq)) - file_of(piecesq));

    return (225 * piececol) + relative;
}

static int nnue_to_relative_psqt(int encoded) {

    /// Given a value [0, 28044], which encodes a (King Sq, Piece-Col, Piece Sq),
    /// compute the relative index mapping of [0, 768] which is the encoded form
    /// of (Piece-Col, Piece Sq).

    const int piecesq   = (encoded % 64);       // Enc = ( 1 * Piece Square )
    const int piececol  = (encoded % 768) / 64; //     + ( 64 * Piece-Col   )
    //    int kingsq    = (encoded / 768);      //     + ( 768 * King Sq    )

    return (64 * piececol) + piecesq;
}

/// Extra helpers for the Architecture

Gradient *L0Gradient;
pthread_mutex_t *L0Locks;

void init_architecture(Network *nn) {

    L0Gradient = create_gradient(nn);
    L0Locks = malloc(sizeof(pthread_mutex_t) * 24576);
    for (int i = 0; i < 24576; i++) pthread_mutex_init(&L0Locks[i], NULL);
}

/// Implementation of the Architecture interface

void insert_indices(bool *array, Sample *sample) {

    int inputs[6];
    uint64_t bb = sample->occupied;

    for (int i = 0; bb != 0ull; i++) {
        compute_inputs(sample, i, poplsb(&bb), inputs);
        for (int j = 0; j < 6; j++) array[inputs[j]] = true;
    }
}

void input_transform(const Sample *sample, const Matrix *matrix, const Vector *bias, Vector *output) {

    uint64_t bb = sample->occupied;
    int inputs[32][2], popcnt = 0, seg1_head = 0, seg2_head = matrix->cols;

    for (popcnt = 0; bb != 0ull; popcnt++)
        compute_real_inputs(sample, popcnt, poplsb(&bb), inputs[popcnt]);

    for (int i = 0; i < bias->length; i++) {
        output->values[seg1_head + i] = bias->values[i];
        output->values[seg2_head + i] = bias->values[i];
    }

    __m256* seg1 = (__m256*) &output->values[seg1_head];
    __m256* seg2 = (__m256*) &output->values[seg2_head];

    for (int i = 0; i < popcnt; i++) {

        __m256* inp1 = (__m256*) &matrix->values[inputs[i][0] * matrix->cols];
        __m256* inp2 = (__m256*) &matrix->values[inputs[i][1] * matrix->cols];

        for (int j = 0; j < matrix->cols / 8; j += 4) {
            seg1[j+0] = _mm256_add_ps(seg1[j+0], inp1[j+0]);
            seg1[j+1] = _mm256_add_ps(seg1[j+1], inp1[j+1]);
            seg1[j+2] = _mm256_add_ps(seg1[j+2], inp1[j+2]);
            seg1[j+3] = _mm256_add_ps(seg1[j+3], inp1[j+3]);
        }

        for (int j = 0; j < matrix->cols / 8; j += 4) {
            seg2[j+0] = _mm256_add_ps(seg2[j+0], inp2[j+0]);
            seg2[j+1] = _mm256_add_ps(seg2[j+1], inp2[j+1]);
            seg2[j+2] = _mm256_add_ps(seg2[j+2], inp2[j+2]);
            seg2[j+3] = _mm256_add_ps(seg2[j+3], inp2[j+3]);
        }
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

    if (batch->indices[idx] >= 24576)
        for (int i = start; i < start + nn->weights[0]->cols; i += 8)
            avx2_update_weights(opt, nn, grads, 0, i, age);

    if (batch->indices[idx] < 24576)
        for (int i = start; i < start + nn->weights[0]->cols; i += 64)
            avx2_update_8x8(opt, nn, &L0Gradient, 0, i, age);
}

void export_network(Network *nn, char *fname) {

    FILE *fout = fopen("exported.nn", "wb");
    load_network(nn, fname);

    {
        #define CLAMP1KB(f)   ((f) > 2048 ? 2048 : ((f) < -2048 ? -2048 : (f)))
        #define QUANTIN16B(f) ((int16_t) CLAMP1KB(roundf(64.0 * (f))))
        #define QUANTIN16W(f) ((int16_t) CLAMP1KB(roundf(64.0 * (f))))

        const int layer = 0;
        const int rows  = nn->weights_t[layer]->rows;
        const int cols  = nn->weights_t[layer]->cols;

        int16_t *biases  = malloc(sizeof(int16_t) * cols);
        int16_t *weights = malloc(sizeof(int16_t) * rows * cols);

        for (int i = 0; i < cols; i++)
            biases[i] = QUANTIN16B(nn->biases[layer]->values[i]);

        for (int i = 0; i < rows * cols; i++)
            weights[i] = QUANTIN16W(nn->weights_t[layer]->values[i]);

        fwrite(biases, sizeof(int16_t), cols, fout);
        fwrite(weights, sizeof(int16_t), rows * cols, fout);
        free(biases); free(weights);

        #undef CLAMP1KB
        #undef QUANTIN16B
        #undef QUANTIN16W
    }

    {
        #define CLAMP8(x)   ((x) > 127 ? 127 : ((x) < -128 ? -128 : (x)))
        #define QUANT32B(f) ((int32_t) (roundf(32.0 * (f))))
        #define QUANT8W(f)  ((int8_t ) (CLAMP8((int) roundf(32.0 * (f)))))

        const int layer = 1;
        const int rows  = nn->weights[layer]->rows;
        const int cols  = nn->weights[layer]->cols;

        int32_t *biases  = malloc(sizeof(int32_t) * cols);
        int8_t  *weights = malloc(sizeof(int8_t ) * rows * cols);

        for (int i = 0; i < cols; i++)
            biases[i] = QUANT32B(nn->biases[layer]->values[i]);

        for (int i = 0; i < rows * cols; i++)
            weights[i] = QUANT8W(nn->weights[layer]->values[i]);

        fwrite(biases, sizeof(int32_t), cols, fout);
        fwrite(weights, sizeof(int8_t), rows * cols, fout);
        free(biases); free(weights);

        #undef CLAMP8
        #undef QUANT32B
        #undef QUANT8W
    }

    for (int layer = 2; layer < nn->layers; layer++) {

        const int rows = nn->weights[layer]->rows;
        const int cols = nn->weights[layer]->cols;

        fwrite(nn->biases[layer]->values, sizeof(float), cols, fout);
        fwrite(nn->weights[layer]->values, sizeof(float), rows * cols, fout);
    }

    fclose(fout);
}

void collapse_input_layer(Network *nn) {

    const int rows = nn->weights_t[0]->rows = 24576;
    const int cols = nn->weights_t[0]->cols;

    #pragma omp parallel for schedule(static) num_threads(NTHREADS)
    for (int i = 0; i < rows; i++) {

        const int offset  = cols * i;
        const int augoff1 = cols * (rows + 0    + nnue_to_relative_kmap(i));
        const int augoff2 = cols * (rows + 2700 + nnue_to_relative_psqt(i));

        for (int j = 0; j < cols; j++)
            nn->weights_t[0]->values[offset+j] = nn->weights[0]->values[offset+j]
                                               + nn->weights[0]->values[augoff1+j]
                                               + nn->weights[0]->values[augoff2+j];
    }
}
