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

#include <stdbool.h>
#include <stdint.h>

#include "../types.h"

typedef struct Sample {
    uint64_t occupied;   // 8-byte occupancy bitboard ( All Pieces )
    int16_t  eval;       // 2-byte int for the target evaluation
    uint8_t  result;     // 1-byte int for result. { L=0, D=1, W=2 }
    uint8_t  packed[16]; // 1-byte int per two pieces
} Sample;

void init_architecture(Network *nn);

void insert_indices(bool *array, Sample *sample);
void input_transform(const Sample *sample, const Matrix *matrix, const Vector *bias, Vector *output);
void apply_backprop_input(Network *nn, Evaluator *eval, Gradient *grad, Sample *sample, float *dlossdz);
void update_input_weights(Optimizer *opt, Network *nn, Gradient **grads, Batch *batch, int idx, int age);
