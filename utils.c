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

#include "utils.h"
#include "types.h"
#include "trainer.h" // TODO: Remove Me

/// Vector Functions

Vector *create_vector(int length) {
    Vector *vector = align_malloc(sizeof(Vector));
    *vector = (Vector) { length, align_malloc(length * sizeof(float)) };
    memset(vector->values, 0, sizeof(float) * vector->length);
    return vector;
}

void delete_vector(Vector *vector) {
    align_free(vector->values); align_free(vector);
}

void set_vector(Vector *vector, float *values) {
    memcpy(vector->values, values, sizeof(float) * vector->length);
}

void zero_vector(Vector *vector) {
    memset(vector->values, 0, sizeof(float) * vector->length);
}

/// Matrix Functions

Matrix *create_matrix(int rows, int cols) {
    Matrix *matrix = align_malloc(sizeof(Matrix));
    *matrix = (Matrix) { rows, cols, align_malloc(rows * cols * sizeof(float)) };
    memset(matrix->values, 0, sizeof(float) * matrix->rows * matrix->cols);
    return matrix;
}

void delete_matrix(Matrix *matrix) {
    align_free(matrix->values); align_free(matrix);
}

void zero_matrix(Matrix *matrix) {
    memset(matrix->values, 0, sizeof(float) * matrix->rows * matrix->cols);
}

/// Evaluator Functions

Evaluator *create_evaluator(Network *nn) {

    Evaluator *eval   = malloc(sizeof(Evaluator));
    eval->unactivated = malloc(sizeof(Vector*) * nn->layers);
    eval->activated   = malloc(sizeof(Vector*) * nn->layers);
    eval->layers      = nn->layers;

    for (int i = 0; i < eval->layers; i++) {

        const int length = i + 1 == eval->layers
                         ? nn->biases[i]->length
                         : nn->weights[i+1]->rows;

        eval->unactivated[i] = create_vector(length);
        eval->activated[i]   = create_vector(length);
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

/// Gradient Functions

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

/// Optimizer Functions

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

/// Chess Utility Functions

int getlsb(uint64_t bb) {
    assert(bb);  // lsb(0) is undefined
    return __builtin_ctzll(bb);
}

int poplsb(uint64_t *bb) {
    int lsb = getlsb(*bb);
    *bb &= *bb - 1;
    return lsb;
}

int file_of(int sq) {
    return sq % 8;
}

int rank_of(int sq) {
    return sq / 8;
}

int square(int rank, int file) {
    return rank * 8 + file;
}

int relative_rank_of(int colour, int sq) {
    return colour == WHITE ? rank_of(sq) : 7 - rank_of(sq);
}

int relative_square(int colour, int sq) {
    return square(relative_rank_of(colour, sq), file_of(sq));
}

int sq64_to_sq32(int sq) {

    static const int LUT[] = {
         3,  2,  1,  0,  0,  1,  2,  3,
         7,  6,  5,  4,  4,  5,  6,  7,
        11, 10,  9,  8,  8,  9, 10, 11,
        15, 14, 13, 12, 12, 13, 14, 15,
        19, 18, 17, 16, 16, 17, 18, 19,
        23, 22, 21, 20, 20, 21, 22, 23,
        27, 26, 25, 24, 24, 25, 26, 27,
        31, 30, 29, 28, 28, 29, 30, 31,
    };

    return LUT[sq];
}

int sq32_to_sq64(int sq) {

    static const int LUT[] = {
         4,  5,  6,  7, 12, 13, 14, 15,
        20, 21, 22, 23, 28, 29, 30, 31,
        36, 37, 38, 39, 44, 45, 46, 47,
        52, 53, 54, 55, 60, 61, 62, 63,
    };

    return LUT[sq];
}

int queen_side_sq(int sq) {
    return (0x0F0F0F0F0F0F0F0FULL >> sq) & 1;
}

int mirror_square(int sq) {
    return square(rank_of(sq), 7 - file_of(sq));
}

/// Operating System Definitions

void* align_malloc(size_t size) {

    #if defined(_WIN32) || defined(_WIN64)
        return _mm_malloc(size, 64);
    #else
        void *mem; return posix_memalign(&mem, 64, size) ? NULL : mem;
    #endif
}

void align_free(void *ptr) {

    #if defined(_WIN32) || defined(_WIN64)
        _mm_free(ptr);
    #else
        free(ptr);
    #endif
}

double get_time_point() {

    #if defined(_WIN32) || defined(_WIN64)
        return (double)(GetTickCount());
    #else
        struct timeval tv;
        double secsInMilli, usecsInMilli;

        gettimeofday(&tv, NULL);
        secsInMilli = ((double)tv.tv_sec) * 1000;
        usecsInMilli = tv.tv_usec / 1000;

        return secsInMilli + usecsInMilli;
    #endif
}
