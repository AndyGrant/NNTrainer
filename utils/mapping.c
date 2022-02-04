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
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct Sample {
    uint64_t occupied;   // 8-byte occupancy bitboard ( No Kings )
    int16_t  eval;       // 2-byte int for the target evaluation
    uint8_t  ply;        // 1-byte int for the current move ply
    uint8_t  plies;      // 1-byte int for the total game length
    uint8_t  result;     // 1-byte int for result. { L=0, D=1, W=2 }
    uint8_t  turn;       // 1-byte int for the side-to-move flag
    uint8_t  wking;      // 1-byte int for the White King Square
    uint8_t  bking;      // 1-byte int for the Black King Square
    uint8_t  packed[15]; // 1-byte int per two non-King pieces
} Sample;

static float sigmoid(float x, float K) {
    return 1.0 / (1.0 + expf(-K * x));
}

static float mse_loss(int eval, float wdl, float K) {
    return powf(sigmoid(eval, K) - wdl, 2.0);
}

static float entire_mse_loss(int *evals, float *wdls, float K, int samples) {

    double total_loss = 0.0;

    #pragma omp parallel shared(total_loss)
    #pragma omp for schedule(static) reduction(+:total_loss)
    for (int i = 0; i < samples; i++)
        total_loss += mse_loss(evals[i], wdls[i], K);

    return (float)(total_loss / samples);
}

static void compute_optimal_K(int *evals, float *wdls, int samples) {

    double start = 0.0, end = 5.0, step = 1.0, curr = 0.0, error;
    double best = entire_mse_loss(evals, wdls, start, samples);

    for (int i = 0; i < 8; i++) {

        curr = start - step;
        while (curr < end) {
            curr = curr + step;
            error = entire_mse_loss(evals, wdls, curr / 400.0, samples);
            if (error <= best) { best = error; start = curr; }
        }

        printf("Iteration [%d] K = [%.9f] E = [%.9f]\n", i, start, best);
        fflush(stdout);

        end   = start + step;
        start = start - step;
        step  = step  / 10.0;
    }
}

int main(int argc, char **argv) {

    const int ChunkSize  = 1024 * 1024;
    const int SampleSize = 1024 * 1024 * 256;
    const char* FileName = "../s15.training.nndata";

    int *evals  = malloc(sizeof(int  ) * SampleSize);
    float *wdls = malloc(sizeof(float) * SampleSize);

    FILE *fin = fopen(FileName, "rb");
    Sample *samples = malloc(sizeof(Sample) * ChunkSize);

    for (int i = 0; i < SampleSize; i+= ChunkSize) {

        fread(samples, sizeof(Sample), ChunkSize, fin);

        for (int j = 0; j < ChunkSize; j++) {
            evals[i+j] = samples[j].eval;
            wdls[i+j]  = samples[j].result / 2.0;
        }
    }

    compute_optimal_K(evals, wdls, SampleSize);
}
