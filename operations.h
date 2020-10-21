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

#include "activate.h"
#include "matrix.h"
#include "vector.h"

void add_array_to_vector(Vector *vector, float *addends);
void add_array_mul_vector_to_matrix(Matrix *matrix, float *mulends, Vector *vector);
void activate_layer(Vector *input, Vector *output, Activation func);

void set_vector_vec_mul_mat(float *output, float *vec, Matrix *mat);
void mul_vector_func_of_vec(float *delta, Vector *vec, float (*func)(float));