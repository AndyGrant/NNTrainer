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
#include <stdio.h>
#include <stdlib.h>

#include "types.h"
#include "config.h"

typedef struct Network {
    int layers;
    Matrix **weights;
    Vector **biases;
    Activation *activations;
    BackProp *backprops;
} Network;

Network *create_network(int length, const Layer *layers);

void delete_network(Network *nn);
void randomize_network(Network *nn);
void save_network(Network *nn, const char *fname);
void load_network(Network *nn, const char *fname);

/**************************************************************************************************************/

#if NN_TYPE == NORMAL

typedef struct Sample {
    uint64_t occupied;   // 8-byte occupancy bitboard ( All Pieces )
    int16_t  eval;       // 2-byte int for the target evaluation
    uint8_t  result;     // 1-byte int for result. { L=0, D=1, W=2 }
    uint8_t  packed[16]; // 1-byte int per two pieces
} Sample;

#elif NN_TYPE == HALFKP

typedef struct Sample {
    uint64_t occupied;   // 8-byte occupancy bitboard ( No Kings )
    int16_t  eval;       // 2-byte int for the target evaluation
    uint8_t  result;     // 1-byte int for result. { L=0, D=1, W=2 }
    uint8_t  turn;       // 1-byte int for the side-to-move flag
    uint8_t  wking;      // 1-byte int for the White King Square
    uint8_t  bking;      // 1-byte int for the Black King Square
    uint8_t  packed[15]; // 1-byte int per two non-King pieces
} Sample;

#endif

Sample *load_samples(const char *fname, int length);
void load_sample(FILE *fin, Sample *sample);

void update_network(Optimizer *opt, Network *nn, Gradient **grads, Batch *batch);
