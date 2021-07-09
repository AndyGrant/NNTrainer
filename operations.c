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

#include <pthread.h>

#include "config.h"
#include "evaluator.h"
#include "gradient.h"
#include "matrix.h"
#include "operations.h"
#include "trainer.h"
#include "vector.h"


#if NN_TYPE == NORMAL

int compute_input(const Sample *sample, int index, int square) {

    #define normal_encode(c, pt, sq) (64 * ((6 * (c)) + (pt)) + sq)

    int piece  = nibble_decode(index, sample->packed) % 8;
    int colour = nibble_decode(index, sample->packed) / 8;

    return normal_encode(colour, piece, square);
}

#endif

#if NN_TYPE == HALFKP

extern pthread_mutex_t *L0Locks;

extern Gradient *L0Gradient;

static int file_of(int sq) { return sq % 8; }

static int rank_of(int sq) { return sq / 8; }

static int square(int rank, int file) { return rank * 8 + file; }

static int relative_rank_of(int colour, int sq) {
    return colour == WHITE ? rank_of(sq) : 7 - rank_of(sq);
}

static int relative_square(int colour, int sq) {
    return square(relative_rank_of(colour, sq), file_of(sq));
}

void compute_inputs(const Sample *sample, int index, int square, int *inputs) {

    #define halfkp_encode(ksq, cr, pt, sq) (640 * (ksq) + 64 * (5 * (cr) + (pt)) + sq)

    int stmk  = sample->turn == WHITE ? sample->wking : sample->bking;
    int nstmk = sample->turn == WHITE ? sample->bking : sample->wking;

    int sksq  = relative_square( sample->turn,  stmk);
    int nsksq = relative_square(!sample->turn, nstmk);

    int srelsq  = relative_square( sample->turn, square);
    int nsrelsq = relative_square(!sample->turn, square);

    int piece  = nibble_decode(index, sample->packed) % 8;
    int colour = nibble_decode(index, sample->packed) / 8;

    int saug  = 15 * (7 + rank_of( sksq) - rank_of( srelsq)) + (7 + file_of( sksq) - file_of( srelsq));
    int nsaug = 15 * (7 + rank_of(nsksq) - rank_of(nsrelsq)) + (7 + file_of(nsksq) - file_of(nsrelsq));

    inputs[0] = (64 * 10 * sksq ) + (64 * (5 * (colour == sample->turn) + piece)) + srelsq;
    inputs[1] = (64 * 10 * nsksq) + (64 * (5 * (colour != sample->turn) + piece)) + nsrelsq;

    inputs[2] = 40960 + (225 * (5 * (colour == sample->turn) + piece)) + saug;
    inputs[3] = 40960 + (225 * (5 * (colour != sample->turn) + piece)) + nsaug;

    inputs[4] = 40960 + 2250 + 64 * (5 * (colour == sample->turn) + piece) + srelsq;
    inputs[5] = 40960 + 2250 + 64 * (5 * (colour != sample->turn) + piece) + nsrelsq;

    #undef halfkp_encode
}

int nnue_to_relative_kmap(int encoded) {

    /// Given a value [0, 40960], which encodes a (King Sq, Piece-Col, Piece Sq),
    /// compute the relative index mapping of [0, 2250] which is the encoded form
    /// of (Piece-Col, Rankwise-distance, Filewise-distance).

    const int piecesq   = (encoded % 64);       // Enc = (1 * Piece Square  )
    const int piececol  = (encoded % 640) / 64; //     + (64 * Piece-Col    )
    const int kingsq    = (encoded / 640);      //     + (640 * King Sq     )

    const int relative  = 15 * (7 + rank_of(kingsq) - rank_of(piecesq))
                             + (7 + file_of(kingsq) - file_of(piecesq));

    return (225 * piececol) + relative;
}

int nnue_to_relative_psqt(int encoded) {

    /// Given a value [0, 40960], which encodes a (King Sq, Piece-Col, Piece Sq),
    /// compute the relative index mapping of [0, 640] which is the encoded form
    /// of (Piece-Col, Piece Sq).

    const int piecesq   = (encoded % 64);       // Enc = (1 * Piece Square  )
    const int piececol  = (encoded % 640) / 64; //     + (64 * Piece-Col    )
    //    int kingsq    = (encoded / 640);      //     + (640 * King Sq     )

    return (64 * piececol) + piecesq;
}

#endif


void add_array_to_vector(Vector *vector, const float *addends) {
    for (int i = 0; i < vector->length; i++)
        vector->values[i] += addends[i];
}

void add_array_mul_vector_to_matrix(Matrix *matrix, const float *mulends, const Vector *vector) {
    for (int i = 0; i < matrix->rows; i++)
        for (int j = 0; j < matrix->cols; j++)
            matrix->values[i * matrix->cols + j] += mulends[j] * vector->values[i];
}

void set_matrix_dot_array_to_array(float *output, const Matrix *matrix, const float *dotends) {

    for (int i = 0; i < matrix->rows; i++) {
        output[i] = 0.0;
        for (int j = 0; j < matrix->cols; j++)
            output[i] += dotends[j] * matrix->values[i * matrix->cols + j];
    }
}


void input_transform(const Sample *sample, const Matrix *matrix, const Vector *bias, Vector *output) {

#if NN_TYPE == NORMAL

    set_vector(output, bias->values);

    uint64_t bb = sample->occupied;

    for (int i = 0; bb != 0ull; i++) {

        int input = compute_input(sample, i, poplsb(&bb));

        for (int j = 0; j < matrix->cols; j++)
            output->values[j] += matrix->values[input * matrix->cols + j];
    }

#elif NN_TYPE == HALFKP

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

#endif

}

void affine_transform(const Vector *vector, const Matrix *matrix, const Vector *bias, Vector *output) {

    set_vector(output, bias->values);

    for (int i = 0; i < matrix->rows; i++)
        for (int j = 0; j < matrix->cols; j++)
            output->values[j] += vector->values[i] * matrix->values[i * matrix->cols + j];
}

void evaluate_network(const Network *nn, Evaluator *eval, const Sample *sample) {

    {
        Vector *outputs   = eval->unactivated[0];
        Vector *activated = eval->activated[0];

        input_transform(sample, nn->weights[0], nn->biases[0], outputs);
        nn->activations[0](outputs, activated);
    }

    for (int layer = 1; layer < nn->layers; layer++) {

        Vector *inputs    = eval->activated[layer-1];
        Vector *outputs   = eval->unactivated[layer];
        Vector *activated = eval->activated[layer];

        affine_transform(inputs, nn->weights[layer], nn->biases[layer], outputs);
        nn->activations[layer](outputs, activated);
    }
}


void build_backprop_grad(Network *nn, Evaluator *eval, Gradient *grad, Sample *sample) {

    const Vector *outputs = eval->activated[nn->layers-1];
    float dlossdz[outputs->length];

    LOSSPROP_FUNC(sample, outputs, dlossdz);
    apply_backprop(nn, eval, grad, sample, dlossdz, nn->layers-1);
}

void apply_backprop(Network *nn, Evaluator *eval, Gradient *grad, Sample *sample, float *dlossdz, int layer) {

    if (layer == 0)
        apply_backprop_input(nn, eval, grad, sample, dlossdz);

    else {

        nn->backprops[layer](dlossdz, eval->unactivated[layer]);
        add_array_to_vector(grad->biases[layer], dlossdz);
        add_array_mul_vector_to_matrix(grad->weights[layer], dlossdz, eval->activated[layer-1]);

        float dlossdz_d1[grad->weights[layer]->rows];
        set_matrix_dot_array_to_array(dlossdz_d1, nn->weights[layer], dlossdz);
        apply_backprop(nn, eval, grad, sample, dlossdz_d1, layer-1);
    }
}

void apply_backprop_input(Network *nn, Evaluator *eval, Gradient *grad, Sample *sample, float *dlossdz) {

#if NN_TYPE == NORMAL

    uint64_t bb = sample->occupied;

    nn->backprops[0](dlossdz, eval->unactivated[0]);
    add_array_to_vector(grad->biases[0], dlossdz);

    for (int i = 0; bb != 0ull; i++) {

        int index = compute_input(sample, i, poplsb(&bb));

        for (int j = 0; j < grad->weights[0]->cols; j++)
            grad->weights[0]->values[index * grad->weights[0]->cols + j] += dlossdz[j];
    }

#elif NN_TYPE == HALFKP

    uint64_t bb = sample->occupied;
    int inputs[6], seg1_head = 0, seg2_head = grad->weights[0]->cols;

    nn->backprops[0](dlossdz, eval->unactivated[0]);
    for (int i = 0; i < grad->biases[0]->length; i++)
        grad->biases[0]->values[i] += dlossdz[seg1_head+i] + dlossdz[seg2_head+i];

    for (int i = 0; bb != 0ull; i++) {

        compute_inputs(sample, i, poplsb(&bb), inputs);

        pthread_mutex_lock(&L0Locks[inputs[0]]);
        for (int j = 0; j < grad->weights[0]->cols; j++)
            L0Gradient->weights[0]->values[inputs[0] * grad->weights[0]->cols + j] += dlossdz[seg1_head + j];
        pthread_mutex_unlock(&L0Locks[inputs[0]]);

        pthread_mutex_lock(&L0Locks[inputs[1]]);
        for (int j = 0; j < grad->weights[0]->cols; j++)
            L0Gradient->weights[0]->values[inputs[1] * grad->weights[0]->cols + j] += dlossdz[seg2_head + j];
        pthread_mutex_unlock(&L0Locks[inputs[1]]);

        for (int j = 0; j < grad->weights[0]->cols; j++) {
            grad->weights[0]->values[inputs[2] * grad->weights[0]->cols + j] += dlossdz[seg1_head + j];
            grad->weights[0]->values[inputs[3] * grad->weights[0]->cols + j] += dlossdz[seg2_head + j];
            grad->weights[0]->values[inputs[4] * grad->weights[0]->cols + j] += dlossdz[seg1_head + j];
            grad->weights[0]->values[inputs[5] * grad->weights[0]->cols + j] += dlossdz[seg2_head + j];
        }
    }

#endif

}
