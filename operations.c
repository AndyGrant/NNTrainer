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

#include "matrix.h"
#include "vector.h"
#include "operations.h"

void add_array_to_vector(Vector *vector, float *addends) {
    for (int i = 0; i < vector->length; i++)
        vector->values[i] += addends[i];
}

void add_array_mul_vector_to_matrix(Matrix *matrix, float *mulends, Vector *vector) {
    for (int i = 0; i < matrix->rows; i++)
        for (int j = 0; j < matrix->cols; j++)
            matrix->values[i * matrix->cols + j] += mulends[j] * vector->values[i];
}

void activate_layer(Vector *input, Vector *output, Activation func) {
    for (int i = 0; i < input->length; i++)
        output->values[i] = func(input->values[i]);
}


void set_vector_vec_mul_mat(float *output, float *vec, Matrix *mat) {

    for (int i = 0; i < mat->rows; i++) {
        output[i] = 0.0;
        for (int j = 0; j < mat->cols; j++)
            output[i] += vec[j] * mat->values[i * mat->cols + j];
    }
}

void mul_vector_func_of_vec(float *delta, Vector *vec, float (*func)(float)) {

    for (int i = 0; i < vec->length; i++)
        delta[i] *= func(vec->values[i]);
}
