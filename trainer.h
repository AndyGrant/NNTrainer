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

Sample *load_samples(const char *fname, int length);

void update_network(Optimizer *opt, Network *nn, Gradient **grads, Batch *batch);
