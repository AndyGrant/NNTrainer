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

static void create_batch(Batch *batch, Sample *start, int batch_size) {

    batch->start  = start;
    batch->inputs = 0;

    bool array[MAX_INPUTS] = {0};
    for (int i = 0; i < batch_size; i++)
        insert_indices(array, &start[i]);

    for (int i = 0; i < MAX_INPUTS; i++)
        batch->inputs += array[i] == true;

    batch->indices = malloc(sizeof(uint16_t) * batch->inputs);
    for (int i = 0, j = 0; i < MAX_INPUTS;  i++)
        if (array[i]) batch->indices[j++] = i;
}

Batch *create_batches(Sample *samples, int nsamples, int batch_size) {

    double start = get_time_point(), elapsed;
    int completed = 0, batch_count = nsamples / batch_size;

    Batch *batches = malloc(sizeof(Batch) * batch_count);
    printf("Performing Batch List Optimizaion (Batch = %d)\n", batch_size);

    #pragma omp parallel for schedule(static) num_threads(NTHREADS) shared(completed)
    for (int i = 0; i < nsamples / batch_size; i++) {

        create_batch(&batches[i], &samples[i * batch_size], batch_size);

        if (++completed % 512 == 0) {
            elapsed = (get_time_point() - start) / 1000.0;
            printf("\r[%8.3fs] Created %d of %d Batch Lists", elapsed, completed, batch_count);
        }
    }

    elapsed = (get_time_point() - start) / 1000.0;
    printf("\r[%8.3fs] Created %d of %d Batch Lists", elapsed, batch_count, batch_count);

    float saved = 0.0;
    for (int i = 0; i < batch_count; i++)
        saved += batches[i].inputs / (float) MAX_INPUTS;
    saved = 100.0 - (saved * 100.0 / batch_count);
    printf("\nAverage savings of %.2f%%\n\n", saved);

    return batches;
}
