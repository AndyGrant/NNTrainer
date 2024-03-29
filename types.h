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

#define PSQBB     0 // Normal PSQBB Networks with a single Input set
#define HALFKP    1 // Networks with 2xHalfs via Shogi-King-Piece encoding
#define MIRRORHKP 2 // HALFKP, with the King mirrored into the King Side
#define MIRRORHKA 3 // MIRRORHKP, but include the Kings in the piece sets

enum { WHITE, BLACK };

enum { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING };

/// Forward declaration of all structs and types for simplifying includes

typedef struct Batch     Batch;
typedef struct Evaluator Evaluator;
typedef struct Gradient  Gradient;
typedef struct Layer     Layer;
typedef struct Matrix    Matrix;
typedef struct Network   Network;
typedef struct Optimizer Optimizer;
typedef struct Sample    Sample;
typedef struct Vector    Vector;

typedef void  (*Activation) (const Vector*, Vector*);
typedef void  (*BackProp)   (float *dlossdz, const Vector*, const Vector*);
typedef float (*Loss)       (const Sample*, const Vector*);
typedef void  (*LossProp)   (const Sample*, const Vector*, float *);

/// Architecture Design, defined here to avoid recursive includes

typedef struct Layer {
    int inputs, outputs;
    Activation activation;
    BackProp backprop;
} Layer;

/// Compression / Decompression for Packed data

#define nibble_decode(i, A) (((i) % 2) ? (A[(i)/2] & 0xF) : (A[(i)/2]) >> 4)
#define nibble_encode(i, A, cp) (A[(i)/2] |= (((i) % 2) ? (cp) : (cp << 4)))

#if NN_TYPE == PSQBB
    #include "archs/psqbb.h"
#elif NN_TYPE == HALFKP
    #include "archs/halfkp.h"
#elif NN_TYPE == MIRRORHKP
    #include "archs/mirrorhkp.h"
#elif NN_TYPE == MIRRORHKA
    #include "archs/mirrorhka.h"
#endif