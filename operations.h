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

#include <stdint.h>

#include "config.h"
#include "types.h"

void add_array_to_vector(Vector *vector, const float *addends);
void add_array_mul_vector_to_matrix(Matrix *matrix, const float *mulends, const Vector *vector);
void set_matrix_dot_array_to_array(float *output, const Matrix *matrix, const float *dotends);
void affine_transform(const Vector *vector, const Matrix *matrix, const Vector *bias, Vector *output);
void evaluate_network(const Network *nn, Evaluator *eval, const Sample *sample);
void build_backprop_grad(Network *nn, Evaluator *eval, Gradient *grad, Sample *sample);
void apply_backprop(Network *nn, Evaluator *eval, Gradient *grad, Sample *sample, float *dlossdz, int layer);
