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

#include "activate.h"
#include "config.h"
#include "trainer.h"
#include "utils.h"

/// Activation and BackProp function utilities. These all take
/// a float. For each Activation, there is a BackProp function

float relu(float x) {
    return fmaxf(0.0, x);
}

float relu_prime(float x) {
    return x > 0.0 ? 1.0 : 0.0;
}

float clipped_relu(float x) {
    return fmaxf(0.0, fminf(1.0, x));
}

float clipped_relu_prime(float x) {
    return (0.0 < x && x < 1.0) ? 1.0 : 0.0;
}

float sigmoid(float x) {
    return 1.0 / (1.0 + expf(-SIGM_COEFF * x));
}

float sigmoid_prime(float x) {
    float sigm = sigmoid(x);
    return SIGM_COEFF * sigm * (1.0 - sigm);
}

/// Activation functions. These functions are all matching
/// >> typedef void (*Activation) (const Vector*, Vector*);

void activate_relu(const Vector *input, Vector *output) {
    for (int i = 0; i < input->length; i++)
        output->values[i] = relu(input->values[i]);
}

void activate_clipped_relu(const Vector *input, Vector *output) {
    for (int i = 0; i < input->length; i++)
        output->values[i] = clipped_relu(input->values[i]);
}

void activate_sigmoid(const Vector *input, Vector *output) {
    for (int i = 0; i < input->length; i++)
        output->values[i] = sigmoid(input->values[i]);
}

void activate_null(const Vector *input, Vector *output) {
    for (int i = 0; i < input->length; i++)
        output->values[i] = input->values[i];
}

/// BackProp functions. These functions are all matching
/// >> typedef void (*BackProp) (float *dlossdz, const Vector *vector, const Vector *vector);

void backprop_relu(float *dlossdz, const Vector *unactivated, const Vector *activated) {
    (void) activated; // Not useful for computing derivatives
    for (int i = 0; i < unactivated->length; i++)
        dlossdz[i] *= relu_prime(unactivated->values[i]);
}

void backprop_clipped_relu(float *dlossdz, const Vector *unactivated, const Vector *activated) {
    (void) activated; // Not useful for computing derivatives
    for (int i = 0; i < unactivated->length; i++)
        dlossdz[i] *= clipped_relu_prime(unactivated->values[i]);
}

void backprop_sigmoid(float *dlossdz, const Vector *unactivated, const Vector *activated) {
    (void) activated; // Not useful for computing derivatives
    for (int i = 0; i < unactivated->length; i++)
        dlossdz[i] *= sigmoid_prime(unactivated->values[i]);
}

void backprop_null(float *dlossdz, const Vector *unactivated, const Vector *activated) {
    (void) dlossdz; (void) unactivated; (void) activated;
}

/// Loss and LossProp functions. These functions are all matching
/// >> typedef float (*Loss)     (const Sample*, const Vector *outputs);
/// >> typedef void  (*LossProp) (const Sample*, const Vector *outputs, float *dlossdz);

float l2_one_neuron_loss(const Sample *sample, Network *nn, Evaluator *evaluator) {

    const float output = evaluator->activated[nn->layers-1]->values[0];

    const float power = 2.0;

    const float eval_weight = 0.00, result_weight = 1.00;

    const float eval =  sigmoid(sample->eval) * eval_weight
                     + (sample->result / 2.0) * result_weight;

    const float lass_lambda = 1 / (1 << 22);

    float summed_activated = 0;
    for (int i = 0; i < 1536; i++)
        summed_activated += evaluator->activated[0]->values[i] > 0;

    return powf(fabs(eval - output), power) + summed_activated * lass_lambda;
}

void l2_one_neuron_lossprop(const Sample *sample, Network *nn, Evaluator *evaluator, float *dlossdz) {

    const float output = evaluator->activated[nn->layers-1]->values[0];

    const float power = 2.0;

    const float eval_weight = 0.00, result_weight = 1.00;

    const float eval =  sigmoid(sample->eval) * eval_weight
                     + (sample->result / 2.0) * result_weight;

    const float sign = (eval > output) ? 1.0 : -1.0;

    const float loss = power * powf(fabs(eval - output), power - 1.0);

    const float lass_lambda = 1 / (1 << 22);

    float summed_activated = 0;
    for (int i = 0; i < 1536; i++)
        summed_activated += evaluator->activated[0]->values[i] > 0;

    *dlossdz = (loss + summed_activated * lass_lambda) * -sign;
}
