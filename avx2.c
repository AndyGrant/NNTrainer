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

#include <immintrin.h>
#include <math.h>

#include "avx2.h"
#include "config.h"
#include "trainer.h"
#include "types.h"
#include "utils.h"

extern int NTHREADS;
extern uint64_t current_iteration;
extern uint64_t last_touched_iteration[MAX_INPUTS];

void avx2_update_weights(Optimizer *opt, Network *nn, Gradient **grads, int layer, int index, int since_last) {

    const __m256 learnrate    = _mm256_set1_ps(LEARNRATE);
    const __m256 zero         = _mm256_setzero_ps();

    const __m256 beta1_normal = _mm256_set1_ps(pow(BETA_1, since_last));
    const __m256 beta1_minus  = _mm256_mul_ps(learnrate, _mm256_set1_ps(1.0 - BETA_1));

    const __m256 beta2_normal = _mm256_set1_ps(pow(BETA_2, since_last));
    const __m256 beta2_minus  = _mm256_set1_ps(1.0 - BETA_2);

    /// Sum up all of the per-thread Gradients

    __m256 accumulated = _mm256_load_ps(&grads[0]->weights[layer]->values[index]);
    for (int i = 1; i < NTHREADS; i++)
        accumulated = _mm256_add_ps(accumulated, _mm256_load_ps(&grads[i]->weights[layer]->values[index]));

    const __m256 accumulated2 = _mm256_mul_ps(accumulated, accumulated);

    /// Compute and Update the Momentum values for Adam

    const __m256 momentum_d1 = _mm256_load_ps(&opt->momentum->weights[layer]->values[index]);

    const __m256 momentum = _mm256_add_ps(
        _mm256_mul_ps(momentum_d1, beta1_normal),
        _mm256_mul_ps(accumulated, beta1_minus)
    );

    _mm256_store_ps(&opt->momentum->weights[layer]->values[index], momentum);

    /// Compute and Update the Velocty values for Adam

    const __m256 velocity_d1 = _mm256_load_ps(&opt->velocity->weights[layer]->values[index]);

    const __m256 velocity = _mm256_add_ps(
        _mm256_mul_ps(velocity_d1, beta2_normal),
        _mm256_mul_ps(accumulated2, beta2_minus)
    );

    _mm256_store_ps(&opt->velocity->weights[layer]->values[index], velocity);

    /// Compute and Update the Weights

    const __m256 s2 = _mm256_add_ps(_mm256_set1_ps(1e-8), _mm256_sqrt_ps(velocity));
    const __m256 deltas  = _mm256_mul_ps(momentum, _mm256_rcp_ps(s2));
    const __m256 updated = _mm256_sub_ps(_mm256_load_ps(&nn->weights[layer]->values[index]), deltas);

    _mm256_store_ps(&nn->weights[layer]->values[index], updated);

    /// Clear the Gradient's for the next batch

    for (int i = 0; i < NTHREADS; i++)
        _mm256_store_ps(&grads[i]->weights[layer]->values[index], zero);
}

void avx2_update_8x8(Optimizer *opt, Network *nn, Gradient **grads, int layer, int index, int since_last) {

    const __m256 learnrate    = _mm256_set1_ps(LEARNRATE);
    const __m256 zero         = _mm256_setzero_ps();
    const __m256 epsilon      = _mm256_set1_ps(1e-8);

    const __m256 beta1_normal = _mm256_set1_ps(pow(BETA_1, since_last));
    const __m256 beta1_minus  = _mm256_mul_ps(learnrate, _mm256_set1_ps(1.0 - BETA_1));

    const __m256 beta2_normal = _mm256_set1_ps(pow(BETA_2, since_last));
    const __m256 beta2_minus  = _mm256_set1_ps(1.0 - BETA_2);

    __m256* const gradients  = (__m256*) &grads[0]->weights[layer]->values[index];
    __m256* const moments    = (__m256*) &opt->momentum->weights[layer]->values[index];
    __m256* const velocities = (__m256*) &opt->velocity->weights[layer]->values[index];
    __m256* const weights    = (__m256*) &nn->weights[layer]->values[index];

    const __m256 momentum_b0 = _mm256_mul_ps(beta1_minus, gradients[0]);
    const __m256 momentum_b1 = _mm256_mul_ps(beta1_minus, gradients[1]);
    const __m256 momentum_b2 = _mm256_mul_ps(beta1_minus, gradients[2]);
    const __m256 momentum_b3 = _mm256_mul_ps(beta1_minus, gradients[3]);
    const __m256 momentum_b4 = _mm256_mul_ps(beta1_minus, gradients[4]);
    const __m256 momentum_b5 = _mm256_mul_ps(beta1_minus, gradients[5]);
    const __m256 momentum_b6 = _mm256_mul_ps(beta1_minus, gradients[6]);
    const __m256 momentum_b7 = _mm256_mul_ps(beta1_minus, gradients[7]);

    moments[0] = _mm256_fmadd_ps(beta1_normal, moments[0], momentum_b0);
    moments[1] = _mm256_fmadd_ps(beta1_normal, moments[1], momentum_b1);
    moments[2] = _mm256_fmadd_ps(beta1_normal, moments[2], momentum_b2);
    moments[3] = _mm256_fmadd_ps(beta1_normal, moments[3], momentum_b3);
    moments[4] = _mm256_fmadd_ps(beta1_normal, moments[4], momentum_b4);
    moments[5] = _mm256_fmadd_ps(beta1_normal, moments[5], momentum_b5);
    moments[6] = _mm256_fmadd_ps(beta1_normal, moments[6], momentum_b6);
    moments[7] = _mm256_fmadd_ps(beta1_normal, moments[7], momentum_b7);

    const __m256 velocity_b0 = _mm256_mul_ps(beta2_minus, _mm256_mul_ps(gradients[0], gradients[0]));
    const __m256 velocity_b1 = _mm256_mul_ps(beta2_minus, _mm256_mul_ps(gradients[1], gradients[1]));
    const __m256 velocity_b2 = _mm256_mul_ps(beta2_minus, _mm256_mul_ps(gradients[2], gradients[2]));
    const __m256 velocity_b3 = _mm256_mul_ps(beta2_minus, _mm256_mul_ps(gradients[3], gradients[3]));
    const __m256 velocity_b4 = _mm256_mul_ps(beta2_minus, _mm256_mul_ps(gradients[4], gradients[4]));
    const __m256 velocity_b5 = _mm256_mul_ps(beta2_minus, _mm256_mul_ps(gradients[5], gradients[5]));
    const __m256 velocity_b6 = _mm256_mul_ps(beta2_minus, _mm256_mul_ps(gradients[6], gradients[6]));
    const __m256 velocity_b7 = _mm256_mul_ps(beta2_minus, _mm256_mul_ps(gradients[7], gradients[7]));

    velocities[0] = _mm256_fmadd_ps(beta2_normal, velocities[0], velocity_b0);
    velocities[1] = _mm256_fmadd_ps(beta2_normal, velocities[1], velocity_b1);
    velocities[2] = _mm256_fmadd_ps(beta2_normal, velocities[2], velocity_b2);
    velocities[3] = _mm256_fmadd_ps(beta2_normal, velocities[3], velocity_b3);
    velocities[4] = _mm256_fmadd_ps(beta2_normal, velocities[4], velocity_b4);
    velocities[5] = _mm256_fmadd_ps(beta2_normal, velocities[5], velocity_b5);
    velocities[6] = _mm256_fmadd_ps(beta2_normal, velocities[6], velocity_b6);
    velocities[7] = _mm256_fmadd_ps(beta2_normal, velocities[7], velocity_b7);

    const __m256 denom0 = _mm256_add_ps(epsilon, _mm256_sqrt_ps(velocities[0]));
    const __m256 denom1 = _mm256_add_ps(epsilon, _mm256_sqrt_ps(velocities[1]));
    const __m256 denom2 = _mm256_add_ps(epsilon, _mm256_sqrt_ps(velocities[2]));
    const __m256 denom3 = _mm256_add_ps(epsilon, _mm256_sqrt_ps(velocities[3]));
    const __m256 denom4 = _mm256_add_ps(epsilon, _mm256_sqrt_ps(velocities[4]));
    const __m256 denom5 = _mm256_add_ps(epsilon, _mm256_sqrt_ps(velocities[5]));
    const __m256 denom6 = _mm256_add_ps(epsilon, _mm256_sqrt_ps(velocities[6]));
    const __m256 denom7 = _mm256_add_ps(epsilon, _mm256_sqrt_ps(velocities[7]));

    const __m256 delta0 = _mm256_mul_ps(moments[0], _mm256_rcp_ps(denom0));
    const __m256 delta1 = _mm256_mul_ps(moments[1], _mm256_rcp_ps(denom1));
    const __m256 delta2 = _mm256_mul_ps(moments[2], _mm256_rcp_ps(denom2));
    const __m256 delta3 = _mm256_mul_ps(moments[3], _mm256_rcp_ps(denom3));
    const __m256 delta4 = _mm256_mul_ps(moments[4], _mm256_rcp_ps(denom4));
    const __m256 delta5 = _mm256_mul_ps(moments[5], _mm256_rcp_ps(denom5));
    const __m256 delta6 = _mm256_mul_ps(moments[6], _mm256_rcp_ps(denom6));
    const __m256 delta7 = _mm256_mul_ps(moments[7], _mm256_rcp_ps(denom7));

    weights[0] = _mm256_sub_ps(weights[0], delta0);
    weights[1] = _mm256_sub_ps(weights[1], delta1);
    weights[2] = _mm256_sub_ps(weights[2], delta2);
    weights[3] = _mm256_sub_ps(weights[3], delta3);
    weights[4] = _mm256_sub_ps(weights[4], delta4);
    weights[5] = _mm256_sub_ps(weights[5], delta5);
    weights[6] = _mm256_sub_ps(weights[6], delta6);
    weights[7] = _mm256_sub_ps(weights[7], delta7);

    gradients[0] = zero;
    gradients[1] = zero;
    gradients[2] = zero;
    gradients[3] = zero;
    gradients[4] = zero;
    gradients[5] = zero;
    gradients[6] = zero;
    gradients[7] = zero;
}