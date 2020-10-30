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

typedef struct Layer {
    int inputs, outputs;
    Activation activation;
    BackProp backprop;
} Layer;

typedef struct Network {
    int layers;
    Matrix **weights;
    Vector **biases;
    Activation *activations;
    BackProp *backprops;
    Loss loss;
    LossProp lossprop;
    int type;
} Network;

Network *create_network(int length, Layer *layers, Loss loss, LossProp lossprop, int type);

void delete_network(Network *nn);
void randomize_network(Network *nn);
void save_network(Network *nn, char *fname);
void load_network(Network *nn, char *fname);

typedef struct Evaluator {
    Vector **unactivated;
    Vector **activated;
    int layers;
} Evaluator;

Evaluator *create_evaluator(Network *nn);
void delete_evaluator(Evaluator *eval);

/**************************************************************************************************************/

#define MAX_INDICIES 30
#define MAX_INPUTS 40960

#define NSAMPLES (1024*32*16)
#define DATAFILE "nnue.d8"

typedef struct Sample {
    int eval, turn, wking, bking, length;
    uint16_t indices[MAX_INDICIES];
} Sample;

Sample *load_samples(char *fname, int length);
void load_sample(FILE *fin, Sample *sample);

/**************************************************************************************************************/

#define BETA_1 0.9
#define BETA_2 0.999

#define LEARNRATE 0.001
#define BATCHSIZE 1024

typedef struct Optimizer {
    Gradient *momentum;
    Gradient *velocity;
} Optimizer;

Optimizer *create_optimizer(Network *nn);
void delete_optimizer(Optimizer *opt);

void update_network(Optimizer *opt, Network *nn, Gradient **grads, Batch *batch);

/**************************************************************************************************************/