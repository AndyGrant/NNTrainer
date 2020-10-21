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

#include "matrix.h"
#include "vector.h"
#include "operations.h"

#include "trainer.h"

void add_array_to_vector(Vector *vector, float *addends) {
    for (int i = 0; i < vector->length; i++)
        vector->values[i] += addends[i];
}

void add_array_mul_vector_to_matrix(Matrix *matrix, float *mulends, Vector *vector) {
    for (int i = 0; i < matrix->rows; i++)
        for (int j = 0; j < matrix->cols; j++)
            matrix->values[i * matrix->cols + j] += mulends[j] * vector->values[i];
}


void set_vector_vec_mul_mat(float *output, float *vec, Matrix *mat) {

    for (int i = 0; i < mat->rows; i++) {
        output[i] = 0.0;
        for (int j = 0; j < mat->cols; j++)
            output[i] += vec[j] * mat->values[i * mat->cols + j];
    }
}


void input_transform(Sample *sample, Matrix *matrix, Vector *bias, Vector *output) {

    set_vector(output, bias->values);

    for (int i = 0; i < sample->length; i++)
        for (int j = 0; j < matrix->cols; j++)
            output->values[j] += matrix->values[sample->indices[i] * matrix->cols + j];
}

void affine_transform(Vector *vector, Matrix *matrix, Vector *bias, Vector *output) {

    set_vector(output, bias->values);

    for (int i = 0; i < matrix->rows; i++)
        for (int j = 0; j < matrix->cols; j++)
            output->values[j] += vector->values[i] * matrix->values[i * matrix->cols + j];
}

void evaluate_network(Network *nn, Evaluator *eval, Sample *sample) {

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
    float dlossdz[outputs->length];

    nn->lossprop(sample, outputs, dlossdz);
    apply_backprop(nn, eval, grad, sample, dlossdz, nn->layers-1);
}

void apply_backprop(Network *nn, Evaluator *eval, Gradient *grad, Sample *sample, float *delta, int layer) {

    if (layer == 0)
        return apply_backprop_input(nn, eval, grad, sample, delta);

    nn->backprops[layer](delta, eval->unactivated[layer]);
    add_array_to_vector(grad->biases[layer], delta);
    add_array_mul_vector_to_matrix(grad->weights[layer], delta, eval->activated[layer-1]);

    float delta_d1[grad->weights[layer]->rows];
    set_vector_vec_mul_mat(delta_d1, delta, nn->weights[layer]);
    apply_backprop(nn, eval, grad, sample, delta_d1, layer-1);
}

void apply_backprop_input(Network *nn, Evaluator *eval, Gradient *grad, Sample *sample, float *delta) {

    nn->backprops[0](delta, eval->unactivated[0]);
    add_array_to_vector(grad->biases[0], delta);

    for (int i = 0; i < sample->length; i++)
        for (int j = 0; j < grad->weights[0]->cols; j++)
            grad->weights[0]->values[sample->indices[i] * grad->weights[0]->cols + j] += delta[j];
}