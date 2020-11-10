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

#include "trainer.h"
#include "types.h"
#include "vector.h"

typedef struct Evaluator {
    Vector **unactivated;
    Vector **activated;
    int layers;
} Evaluator;

INLINE Evaluator *create_evaluator(Network *nn) {

    Evaluator *eval   = malloc(sizeof(Evaluator));
    eval->unactivated = malloc(sizeof(Vector*) * nn->layers);
    eval->activated   = malloc(sizeof(Vector*) * nn->layers);
    eval->layers      = nn->layers;

#if NN_TYPE == NORMAL

    for (int i = 0; i < eval->layers; i++) {
        eval->unactivated[i] = create_vector(nn->biases[i]->length);
        eval->activated[i]   = create_vector(nn->biases[i]->length);
    }

#elif NN_TYPE == HALFKP

    eval->unactivated[0] = create_vector(2 * nn->biases[0]->length);
    eval->activated[0]   = create_vector(2 * nn->biases[0]->length);

    for (int i = 1; i < eval->layers; i++) {
        eval->unactivated[i] = create_vector(nn->biases[i]->length);
        eval->activated[i]   = create_vector(nn->biases[i]->length);
    }

#endif

    return eval;
}

INLINE void delete_evaluator(Evaluator *eval) {

    for (int i = 0; i < eval->layers; i++) {
        delete_vector(eval->unactivated[i]);
        delete_vector(eval->activated[i]);
    }

    free(eval->unactivated);
    free(eval->activated);
    free(eval);
}
