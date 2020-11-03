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

#include "gradient.h"
#include "trainer.h"
#include "types.h"

#define BETA_1 0.9
#define BETA_2 0.999

typedef struct Optimizer {
    Gradient *momentum;
    Gradient *velocity;
} Optimizer;

INLINE Optimizer *create_optimizer(Network *nn) {
    Optimizer *opt = malloc(sizeof(Optimizer));
    opt->momentum = create_gradient(nn);
    opt->velocity = create_gradient(nn);
    return opt;
}

INLINE void delete_optimizer(Optimizer *opt) {
    free(opt->momentum);
    free(opt->velocity);
    free(opt);
}
