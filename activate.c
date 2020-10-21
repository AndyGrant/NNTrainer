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


float loss_function(float x, float y) {
    return pow(y - x, 2.0);
}

float loss_prime(float x, float y) {
    return -2.0 * (y - x);
}


void activate_layer(Vector *input, Vector *output, Activation func) {
    for (int i = 0; i < input->length; i++)
        output->values[i] = func(input->values[i]);
}