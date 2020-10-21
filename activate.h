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

#include <math.h>

#include "vector.h"

typedef struct Sample Sample; // DELETE ME

#define SIGM_COEFF (3.145 / 400.000)

typedef float (*Activation) (float);
typedef float (*Loss)       (const Sample*, const Vector *outputs);
typedef void  (*BackProp)   (const Sample*, const Vector *outputs, float *dlossdz);

/// Activation functions and deriviatives. These functions
/// are all matching >> typedef float (*Activation) (float);

float relu(float x);
float relu_prime(float x);

float sigmoid(float x);
float sigmoid_prime(float x);

float null_activation(float x);
float null_activation_prime(float x);

float loss_function(float x, float y);
float loss_prime(float x, float y);

/// Loss and BackProp functions. These functions are all matching
/// >> typedef float (*Loss)     (const Sample*, const Vector *outputs);
/// >> typedef void  (*BackProp) (const Sample*, const Vector *outputs, float *dlossdz);

float l2_loss_one_neuron(const Sample *sample, const Vector *outputs);
void l2_loss_one_neuron_backprop(const Sample *sample, const Vector *outputs, float *dlossdz);


/// Not something to add to or to change. Just a generic to apply an Activation
/// to a layer of the Evaluator. { input[x] = func(output[x]) for all x }

void activate_layer(Vector *input, Vector *output, Activation func);