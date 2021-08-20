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

#pragma once

#include <assert.h>
#include <stdalign.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32) || defined(_WIN64)
    #include <windows.h>
#else
    #include <sys/time.h>
#endif

#include "types.h"

#define ALIGN64 alignas(64)

/// Vector Declarations

typedef struct Vector {
    int length;
    float ALIGN64 *values;
} Vector;

Vector *create_vector(int length);
void delete_vector(Vector *vector);
void set_vector(Vector *vector, float *values);
void zero_vector(Vector *vector);

/// Matrix Declarations

typedef struct Matrix {
    int rows, cols;
    float ALIGN64 *values;
} Matrix;

Matrix *create_matrix(int rows, int cols);
void delete_matrix(Matrix *matrix);
void zero_matrix(Matrix *matrix);

/// Evaluator Declarations

typedef struct Evaluator {
    Vector **unactivated;
    Vector **activated;
    int layers;
} Evaluator;

Evaluator *create_evaluator(Network *nn);
void delete_evaluator(Evaluator *eval);

/// Gradient Declarations

typedef struct Gradient {
    Matrix **weights;
    Vector **biases;
    int layers;
} Gradient;

Gradient *create_gradient(Network *nn);
void delete_gradient(Gradient *grad);
void zero_gradient(Gradient *grad);

/// Optimizer Declarations

typedef struct Optimizer {
    Gradient *momentum;
    Gradient *velocity;
} Optimizer;

Optimizer *create_optimizer(Network *nn);
void delete_optimizer(Optimizer *opt);

/// Chess Utility Declarations

int getlsb(uint64_t bb);
int poplsb(uint64_t *bb);
int file_of(int sq);
int rank_of(int sq);
int square(int rank, int file);
int relative_rank_of(int colour, int sq);
int relative_square(int colour, int sq);

// Operating System Function Declarations

void* align_malloc(size_t size);
void align_free(void *ptr);
double get_time_point();
