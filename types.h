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

  You should have received a copy of the GNU General Public License
  GNU General Public License for more details.
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <stdalign.h>

enum { NORMAL, HALFKP };
enum { WHITE, BLACK };

typedef struct Batch     Batch;
typedef struct Evaluator Evaluator;
typedef struct Gradient  Gradient;
typedef struct Layer     Layer;
typedef struct Matrix    Matrix;
typedef struct Network   Network;
typedef struct Optimizer Optimizer;
typedef struct Sample    Sample;
typedef struct Vector    Vector;

typedef void  (*Activation) (Vector*, const Vector*);
typedef void  (*BackProp)   (float *dlossdz, const Vector*);
typedef float (*Loss)       (const Sample*, const Vector*);
typedef void  (*LossProp)   (const Sample*, const Vector*, float *);

#define ALIGN64 alignas(64)
#define INLINE static inline

#include "utils.h"
