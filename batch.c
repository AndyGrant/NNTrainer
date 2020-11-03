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

extern int NTHREADS;

Batch *create_batches(Sample *samples, int nsamples, int batch_size) {

    int completed = 0;
    Batch *batches = malloc(sizeof(Batch) * nsamples / batch_size);

    printf("Performing Batch Optimizaion (Batch = %d)\n", batch_size);
    fflush(stdout);

    #pragma omp parallel for schedule(static) num_threads(NTHREADS) shared(completed)
    for (int i = 0; i < nsamples / batch_size; i++) {
        create_batch(&batches[i], &samples[i * batch_size], batch_size);
        printf("\rCreated %d of %d Batch Lists", ++completed, nsamples / batch_size);
        fflush(stdout);
    }

    float saved = 0.0;
    for (int i = 0; i < nsamples / batch_size; i++)
        saved += batches[i].inputs / (float) MAX_INPUTS;
    saved = 100.0 - (saved * 100.0 / (nsamples / batch_size));

    printf("\nFinished Creating Batch Lists");
    printf("\nAverage savings of %.2f%%\n\n", saved);
    fflush(stdout);

    return batches;
}

void create_batch(Batch *batch, Sample *start, int batch_size) {

    batch->start  = start;
    batch->inputs = 0;

    uint16_t array[MAX_INPUTS];
    for (int i = 0; i < batch_size; i++)
        append_indices(array, &batch->inputs, &start[i]);

    batch->indices = malloc(sizeof(uint16_t) * batch->inputs);
    memcpy(batch->indices, array, sizeof(uint16_t) * batch->inputs);
}

void append_indices(uint16_t *array, int *length, Sample *sample) {

    for (int i = 0; i < sample->length; i++) {

        int seg1_idx, seg2_idx;
        compute_indices(sample, sample->indices[i], &seg1_idx, &seg2_idx);

        append_index(array, length, seg1_idx);
        append_index(array, length, seg2_idx);
    }
}

void append_index(uint16_t *array, int *length, uint16_t index) {

    if (*length == 0 || index > array[*length-1]) {
        array[(*length)++] = index;
        return;
    }

    if (index < array[0]) {

        for (int i = *length; i >= 1; i--)
            array[i] = array[i-1];

        array[0] = index;
        *length = *length + 1;
        return;
    }

    for (int i = 0; i < *length; i++) {

        if (array[i] > index)
            return;

        if (array[i] < index && index < array[i+1]) {

            for (int j = *length; j-1 >= i+1; j--)
                array[j] = array[j-1];

            array[i+1] = index;
            *length = *length + 1;
            return;
        }
    }
}
