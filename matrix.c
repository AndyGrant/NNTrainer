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

#include <stdio.h>
#include <stdlib.h>

#include "matrix.h"

Matrix *create_matrix(int rows, int cols) {
    Matrix *mat = malloc(sizeof(Matrix));
    *mat = (Matrix) { rows, cols, calloc(rows * cols, sizeof(float)) };
    return mat;
}

void delete_matrix(Matrix *mat) {
    free(mat->values); free(mat);
}

void print_matrix(const Matrix *mat) {

    printf("[");

    for (int i = 0; i < mat->rows; i++) {
        printf(i == 0 ? "[ " : " [ ");
        for (int j = 0; j < mat->cols; j++)
            printf(PRINT_FORMAT " ", mat->values[i * mat->cols + j]);
        printf(i == mat->rows - 1 ? "]]\n\n" : "]\n");
    }
}
