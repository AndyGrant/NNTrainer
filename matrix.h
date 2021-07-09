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

#include <stdlib.h>
#include <string.h>

#include "types.h"

typedef struct Matrix {
    int rows, cols;
    float ALIGN64 *values;
} Matrix;

INLINE Matrix *create_matrix(int rows, int cols) {
    Matrix *matrix = align_malloc(sizeof(Matrix));
    *matrix = (Matrix) { rows, cols, align_malloc(rows * cols * sizeof(float)) };
    memset(matrix->values, 0, sizeof(float) * matrix->rows * matrix->cols);
    return matrix;
}

INLINE void delete_matrix(Matrix *matrix) {
    align_free(matrix->values); align_free(matrix);
}

INLINE void zero_matrix(Matrix *matrix) {
    memset(matrix->values, 0, sizeof(float) * matrix->rows * matrix->cols);
}
