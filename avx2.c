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

  You should have received a copy of the GNU General Public License
  GNU General Public License for more details.
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <immintrin.h>

#include "avx2.h"
#include "config.h"
#include "gradient.h"
#include "matrix.h"
#include "optimizer.h"
#include "trainer.h"
#include "types.h"
#include "vector.h"

void avx2_update_weights(Optimizer *opt, Network *nn, Gradient **grads, int layer, int index, int chunks) {

    const __m256 beta1_normal = _mm256_set1_ps(BETA_1);
    const __m256 beta1_minus  = _mm256_set1_ps(1.0 - BETA_1);

    const __m256 beta2_normal = _mm256_set1_ps(BETA_2);
    const __m256 beta2_minus  = _mm256_set1_ps(1.0 - BETA_2);

    const __m256 batchsize    = _mm256_set1_ps(BATCHSIZE);
    const __m256 learnrate    = _mm256_set1_ps(LEARNRATE);

    const __m256 zero         = _mm256_setzero_ps();

    /// Sum up all of the per-thread Gradients

    __m256 partial = _mm256_load_ps(&grads[0]->weights[layer]->values[index]);
    for (int i = 1; i < chunks; i++)
        partial = _mm256_add_ps(partial, _mm256_load_ps(&grads[i]->weights[layer]->values[index]));

    const __m256 accumulated = _mm256_div_ps(partial, batchsize);
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

    const __m256 s1 = _mm256_mul_ps(learnrate, momentum);
    const __m256 s2 = _mm256_add_ps(_mm256_set1_ps(1e-8), _mm256_sqrt_ps(velocity));

    const __m256 deltas  = _mm256_mul_ps(s1, _mm256_rcp_ps(s2));
    const __m256 updated = _mm256_sub_ps(_mm256_load_ps(&nn->weights[layer]->values[index]), deltas);

    _mm256_store_ps(&nn->weights[layer]->values[index], updated);

    /// Clear the Gradient's for the next batch

    for (int i = 0; i < chunks; i++)
        _mm256_store_ps(&grads[i]->weights[layer]->values[index], zero);
}
