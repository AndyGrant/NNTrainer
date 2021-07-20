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

#include <pthread.h>

#include "config.h"
#include "operations.h"
#include "trainer.h"
#include "utils.h"

void add_array_to_vector(Vector *vector, const float *addends) {
    for (int i = 0; i < vector->length; i++)
        vector->values[i] += addends[i];
}

void add_array_mul_vector_to_matrix(Matrix *matrix, const float *mulends, const Vector *vector) {
    for (int i = 0; i < matrix->rows; i++)
        for (int j = 0; j < matrix->cols; j++)
            matrix->values[i * matrix->cols + j] += mulends[j] * vector->values[i];
}

void set_matrix_dot_array_to_array(float *output, const Matrix *matrix, const float *dotends) {

    for (int i = 0; i < matrix->rows; i++) {
        output[i] = 0.0;
        for (int j = 0; j < matrix->cols; j++)
            output[i] += dotends[j] * matrix->values[i * matrix->cols + j];
    }
}

void affine_transform(const Vector *vector, const Matrix *matrix, const Vector *bias, Vector *output) {

    set_vector(output, bias->values);

    for (int i = 0; i < matrix->rows; i++)
        for (int j = 0; j < matrix->cols; j++)
            output->values[j] += vector->values[i] * matrix->values[i * matrix->cols + j];
}

void evaluate_network(const Network *nn, Evaluator *eval, const Sample *sample) {

    {
        Vector *outputs   = eval->unactivated[0];
        Vector *activated = eval->activated[0];

        input_transform(sample, nn->weights[0], nn->biases[0], outputs);
        nn->activations[0](outputs, activated);
    }

    for (int layer = 1; layer < nn->layers; layer++) {

        Vector *inputs    = eval->activated[layer-1];
        Vector *outputs   = eval->unactivated[layer];
        Vector *activated = eval->activated[layer];

        affine_transform(inputs, nn->weights[layer], nn->biases[layer], outputs);
        nn->activations[layer](outputs, activated);
    }
}

void build_backprop_grad(Network *nn, Evaluator *eval, Gradient *grad, Sample *sample) {

    const Vector *outputs = eval->activated[nn->layers-1];
    ALIGN64 float dlossdz[outputs->length];

    LOSSPROP_FUNC(sample, outputs, dlossdz);
    apply_backprop(nn, eval, grad, sample, dlossdz, nn->layers-1);
}

void apply_backprop(Network *nn, Evaluator *eval, Gradient *grad, Sample *sample, float *dlossdz, int layer) {

    if (layer == 0)
        apply_backprop_input(nn, eval, grad, sample, dlossdz);

    else {

        nn->backprops[layer](dlossdz, eval->unactivated[layer], eval->activated[layer]);
        add_array_to_vector(grad->biases[layer], dlossdz);
        add_array_mul_vector_to_matrix(grad->weights[layer], dlossdz, eval->activated[layer-1]);

        ALIGN64 float dlossdz_d1[grad->weights[layer]->rows];
        set_matrix_dot_array_to_array(dlossdz_d1, nn->weights[layer], dlossdz);
        apply_backprop(nn, eval, grad, sample, dlossdz_d1, layer-1);
    }
}
