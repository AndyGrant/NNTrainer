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

typedef struct Vector {
    int length;
    float ALIGN64 *values;
} Vector;

INLINE Vector *create_vector(int length) {
    Vector *vector = align_malloc(sizeof(Vector));
    *vector = (Vector) { length, align_malloc(length * sizeof(float)) };
    memset(vector->values, 0, sizeof(float) * vector->length);
    return vector;
}

INLINE void delete_vector(Vector *vector) {
    align_free(vector->values); align_free(vector);
}

INLINE void set_vector(Vector *vector, float *values) {
    memcpy(vector->values, values, sizeof(float) * vector->length);
}

INLINE void zero_vector(Vector *vector) {
    memset(vector->values, 0, sizeof(float) * vector->length);
}
