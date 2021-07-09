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

#include "matrix.h"
#include "trainer.h"
#include "types.h"
#include "vector.h"

typedef struct Gradient {
    Matrix **weights;
    Vector **biases;
    int layers;
} Gradient;

INLINE Gradient *create_gradient(Network *nn) {

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

INLINE void delete_gradient(Gradient *grad) {

    for (int i = 0; i < grad->layers; i++) {
        delete_matrix(grad->weights[i]);
        delete_vector(grad->biases[i]);
    }

    free(grad->weights);
    free(grad->biases );
    free(grad);
}

INLINE void zero_gradient(Gradient *grad) {
    for (int i = 0; i < grad->layers; i++) {
        zero_matrix(grad->weights[i]);
        zero_vector(grad->biases[i]);
    }
}
