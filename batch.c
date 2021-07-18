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

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "batch.h"
#include "config.h"
#include "operations.h"
#include "trainer.h"
#include "types.h"
#include "utils.h"

extern int NTHREADS;

static void insert_value(uint16_t *arr, int *len, uint16_t value) {

    int i, left = 0, right = *len;

    while (left < right) {
        i = (left + right) / 2;
        if (value < arr[i]) right = i;
        else if (value > arr[i]) left = i + 1;
        else return;
    }

    memmove(arr + left + 1, arr + left, (*len - left) * sizeof(uint16_t));
    arr[left] = value;
    ++*len;
}

static void insert_indices(uint16_t *array, int *length, Sample *sample) {

#if NN_TYPE == NORMAL

    uint64_t bb = sample->occupied;

    for (int i = 0; bb != 0ull; i++)
        insert_value(array, length, compute_input(sample, i, poplsb(&bb)));

#elif NN_TYPE == HALFKP

    int inputs[6];
    uint64_t bb = sample->occupied;

    for (int i = 0; bb != 0ull; i++) {

        compute_inputs(sample, i, poplsb(&bb), inputs);

        for (int j = 0; j < 6; j++)
            insert_value(array, length, inputs[j]);
    }

#endif
}

static void create_batch(Batch *batch, Sample *start, int batch_size) {

    batch->start  = start;
    batch->inputs = 0;

    uint16_t array[MAX_INPUTS];
    for (int i = 0; i < batch_size; i++)
        insert_indices(array, &batch->inputs, &start[i]);

    batch->indices = malloc(sizeof(uint16_t) * batch->inputs);
    memcpy(batch->indices, array, sizeof(uint16_t) * batch->inputs);
}

Batch *create_batches(Sample *samples, int nsamples, int batch_size) {

    int completed = 0;
    Batch *batches = malloc(sizeof(Batch) * nsamples / batch_size);
    printf("Performing Batch List Optimizaion (Batch = %d)\n", batch_size);

    #pragma omp parallel for schedule(static) num_threads(NTHREADS) shared(completed)
    for (int i = 0; i < nsamples / batch_size; i++) {
        create_batch(&batches[i], &samples[i * batch_size], batch_size);
        if (++completed % 64 == 0)
            printf("\rCreated %d of %d Batch Lists", completed, nsamples / batch_size);
    }

    printf("\rCreated %d of %d Batch Lists", nsamples / batch_size, nsamples / batch_size);

    float saved = 0.0;
    for (int i = 0; i < nsamples / batch_size; i++)
        saved += batches[i].inputs / (float) MAX_INPUTS;
    saved = 100.0 - (saved * 100.0 / (nsamples / batch_size));
    printf("\nAverage savings of %.2f%%\n\n", saved);

    return batches;
}
