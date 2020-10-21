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

#include <math.h>

#include "vector.h"
#include "activate.h"
#include "trainer.h"

/// Activation functions and deriviatives. These functions
/// are all matching >> typedef float (*Activation) (float);

float relu(float x) {
    return fmaxf(0.0, x);
}

float relu_prime(float x) {
    return x > 0.0 ? 1.0 : 0.0;
}

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-SIGM_COEFF * x));
}

float sigmoid_prime(float x) {
    float sigm = sigmoid(x);
    return SIGM_COEFF * sigm * (1.0 - sigm);
}

float null_activation(float x) {
    return x;
}

float null_activation_prime(float x) {
    (void) x; return 1.0;
}


/// Loss and BackProp functions. These functions are all matching
/// >> typedef float (*Loss)     (const Sample*, const Vector *outputs);
/// >> typedef void  (*BackProp) (const Sample*, const Vector *outputs, float *dlossdz);

float l2_loss_one_neuron(const Sample *sample, const Vector *outputs) {
    return pow(sample->result - outputs->values[0], 2.0);
}

void l2_loss_one_neuron_backprop(const Sample *sample, const Vector *outputs, float *dlossdz) {
    *dlossdz = -2.0 * (sample->result - outputs->values[0]);
}
