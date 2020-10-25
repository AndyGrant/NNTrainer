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

float relu(float x) {
    return fmaxf(0.0, x);
}

float relu_prime(float x) {
    return x > 0.0 ? 1.0 : 0.0;
}

float sigmoid(float x) {
    return 1.0 / (1.0 + expf(-SIGM_COEFF * x));
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

/// Activation functions. These functions are all matching
/// >> typedef void (*Activation) (Vector*, const Vector*);

void activate_relu(Vector *input, const Vector *output) {
    for (int i = 0; i < input->length; i++)
        output->values[i] = relu(input->values[i]);
}

void activate_sigmoid(Vector *input, const Vector *output) {
    for (int i = 0; i < input->length; i++)
        output->values[i] = sigmoid(input->values[i]);
}

void activate_null(Vector *input, const Vector *output) {
    for (int i = 0; i < input->length; i++)
        output->values[i] = null_activation(input->values[i]);
}

/// BackProp functions. These functions are all matching
/// >>typedef void (*BackProp) (float *dlossdz, const Vector *vector);

void backprop_relu(float *dlossdz, const Vector *vector) {
    for (int i = 0; i < vector->length; i++)
        dlossdz[i] *= relu_prime(vector->values[i]);
}

void backprop_sigmoid(float *dlossdz, const Vector *vector) {
    for (int i = 0; i < vector->length; i++)
        dlossdz[i] *= sigmoid_prime(vector->values[i]);
}

void backprop_null(float *dlossdz, const Vector *vector) {
    for (int i = 0; i < vector->length; i++)
        dlossdz[i] *= null_activation_prime(vector->values[i]);
}

/// Loss and LossProp functions. These functions are all matching
/// >> typedef float (*Loss)     (const Sample*, const Vector *outputs);
/// >> typedef void  (*LossProp) (const Sample*, const Vector *outputs, float *dlossdz);

float l2_one_neuron_loss(const Sample *sample, const Vector *outputs) {
    return powf(sigmoid(sample->eval) - outputs->values[0], 2.0);
}

void l2_one_neuron_lossprob(const Sample *sample, const Vector *outputs, float *dlossdz) {
    *dlossdz = -2.0 * (sigmoid(sample->eval) - outputs->values[0]);
}

// float l2_loss_phased(const Sample *sample, const Vector *outputs) {
//
//     float mg = sample->mgeval + outputs->values[0];
//     float eg = sample->egeval + outputs->values[1];
//
//     float mg_rho = 1.0 - sample->phase / 24.0;
//     float eg_rho = 0.0 + sample->phase / 24.0;
//
//     float xi   = sample->scale / 128.0;
//     float eval = mg * mg_rho + eg * eg_rho * xi + 20.0;
//
//     return powf(sample->eval - sigmoid(eval), 2.0);
// }
//
// void l2_loss_phased_lossprop(const Sample *sample, const Vector *outputs, float *dlossdz) {
//
//     float mg = sample->mgeval + outputs->values[0];
//     float eg = sample->egeval + outputs->values[1];
//
//     float mg_rho = 1.0 - sample->phase / 24.0;
//     float eg_rho = 0.0 + sample->phase / 24.0;
//
//     float xi   = sample->scale / 128.0;
//     float eval = mg * mg_rho + eg * eg_rho * xi + 20.0;
//     float base = -2.0 * sigmoid_prime(eval) * (sample->eval - sigmoid(eval));
//
//     dlossdz[0] = base * mg_rho;
//     dlossdz[1] = base * eg_rho * xi;
// }

