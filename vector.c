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

#include <stdlib.h>
#include <string.h>

#include "vector.h"

Vector *create_vector(int length) {
    Vector *vector = malloc(sizeof(Vector));
    *vector = (Vector) { length, calloc(length, sizeof(float)) };
    return vector;
}

void delete_vector(Vector *vector) {
    free(vector->values); free(vector);
}

void set_vector(Vector *vector, float *values) {
    memcpy(vector->values, values, sizeof(float) * vector->length);
}

void zero_vector(Vector *vector) {
    memset(vector->values, 0, sizeof(float) * vector->length);
}
