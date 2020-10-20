#include <assert.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "trainer.h"

int main() {

    Network *nn = create_network(2, 8, 2, 1);
    print_network(nn);

    Evaluator *eval = create_evaluator(nn);
    Gradient *grad = create_gradient(nn);

    Vector *sample = create_vector(8);
    set_vector(sample, 8, (float[]) { -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0 });
    printf("Input Vector\n");
    print_vector(sample);

    dense_evaluate_network(nn, eval, sample);
    print_evaluator(eval);

    build_backprop_grad(nn, eval, grad, sample, 1.0);

    delete_network(nn);
}

/**************************************************************************************************************/

Vector *create_vector(int length) {

    Vector *vec = malloc(sizeof(Vector));
    vec->length = length;
    vec->values = calloc(length, sizeof(float));

    return vec;
}

void delete_vector(Vector *vec) {
    free(vec->values); free(vec);
}

void print_vector(const Vector *vec) {

    printf("[ ");

    for (int i = 0; i < vec->length; i++)
        printf(PRINT_FORMAT " ", vec->values[i]);

    printf("]\n\n");
}

void set_vector(Vector *vec, int length, float *values) {

    assert(length == vec->length);

    for (int i = 0; i < length; i++)
        vec->values[i] = values[i];
}

void zero_vector(Vector *vec) {

    for (int i = 0; i < vec->length; i++)
        vec->values[i] = 0.0;
}


Matrix *create_matrix(int rows, int cols) {

    Matrix *mat = malloc(sizeof(Vector));
    mat->rows   = rows;
    mat->cols   = cols;
    mat->values = calloc(rows * cols, sizeof(float));

    return mat;
}

void delete_matrix(Matrix *mat) {
    free(mat->values); free(mat);
}

void print_matrix(const Matrix *mat) {

    printf("[");

    for (int i = 0; i < mat->rows; i++) {
        printf(i == 0 ? "[ " : " [ ");
        for (int j = 0; j < mat->cols; j++)
            printf(PRINT_FORMAT " ", mat->values[i * mat->cols + j]);
        printf(i == mat->rows - 1 ? "]]\n\n" : "]\n");
    }
}

void set_matrix_row(Matrix *mat, int row, int cols, float *values) {

    assert(0 <= row && row < mat->rows);
    assert(cols == mat->cols);

    for (int j = 0; j < cols; j++)
        mat->values[row * mat->cols + j] = values[j];

}

void set_matrix_col(Matrix *mat, int col, int rows, float *values) {

    assert(0 <= col && col < mat->cols);
    assert(rows == mat->rows);

    for (int i = 0; i < rows; i++)
        mat->values[i * mat->cols + col] = values[i];

}


void affine_transform(Vector *vector, Matrix *matrix, Vector *output) {

    assert(output->length == matrix->cols);
    assert(vector->length == matrix->rows);

    zero_vector(output);

    for (int i = 0; i < matrix->rows; i++)
        for (int j = 0; j < matrix->cols; j++)
            output->values[j] += vector->values[i] * matrix->values[i * matrix->cols + j];
}

void affine_transform_bias(Vector *vector, Matrix *matrix, Vector *bias, Vector *output) {

    assert(output->length == bias->length);
    assert(output->length == matrix->cols);
    assert(vector->length == matrix->rows);

    set_vector(output, bias->length, bias->values);

    for (int i = 0; i < matrix->rows; i++)
        for (int j = 0; j < matrix->cols; j++)
            output->values[j] += vector->values[i] * matrix->values[i * matrix->cols + j];
}


void add_vector(Vector *vec, float *addends) {
    for (int i = 0; i < vec->length; i++)
        vec->values[i] += addends[i];
}

void add_matrix_vec_mul_vec(Matrix *mat, float *vec1, Vector *vec2) {

    for (int i = 0; i < mat->rows; i++)
        for (int j = 0; j < mat->cols; j++)
            mat->values[i * mat->cols + j] += vec1[j] * vec2->values[i];
}

/**************************************************************************************************************/

Network *create_network(int layers, ...) {

    Network *nn = malloc(sizeof(Network));

    nn->layers  = layers;
    nn->weights = malloc(sizeof(Matrix*) * layers);
    nn->biases  = malloc(sizeof(Vector*) * layers);

    va_list args;
    va_start(args, layers);

    int sizes[layers + 1];
    for (int i = 0; i < layers + 1; i++)
        sizes[i] = va_arg(args, int);

    va_end(args);

    for (int i = 0; i < layers; i++) {
        nn->weights[i] = create_matrix(sizes[i], sizes[i+1]);
        nn->biases[i]  = create_vector(sizes[i+1]);
    }

    randomize_network(nn);

    return nn;
}

void delete_network(Network *nn) {

    for (int i = 0; i < nn->layers; i++) {
        delete_matrix(nn->weights[i]);
        delete_vector(nn->biases[i]);
    }

    free(nn->weights);
    free(nn->biases);
    free(nn);
}

void print_network(Network *nn) {

    for (int i = 0; i < nn->layers; i++) {

        printf("Layer %d Weights (%dx%d)\n", i + 1, nn->weights[i]->rows, nn->weights[i]->cols);
        print_matrix(nn->weights[i]);

        printf("Layer %d Biases (%dx1)\n", i + 1, nn->biases[i]->length);
        print_vector(nn->biases[i]);
    }
}

void randomize_network(Network *nn) {

    #define random_weight() ((rand() % 4))

    for (int i = 0; i < nn->layers; i++) {

        for (int j = 0; j < nn->weights[i]->rows * nn->weights[i]->cols; j++)
            nn->weights[i]->values[j] = random_weight();

        for (int j = 0; j < nn->biases[i]->length; j++)
            nn->biases[i]->values[j] = random_weight();
    }

    #undef random_weight
}


Evaluator *create_evaluator(Network *nn) {

    Evaluator *eval   = malloc(sizeof(Evaluator));
    eval->layers      = nn->layers;
    eval->neurons     = malloc(sizeof(Vector*) * eval->layers);
    eval->activations = malloc(sizeof(Vector*) * eval->layers);

    for (int i = 0; i < eval->layers; i++) {
        eval->neurons[i]     = create_vector(nn->biases[i]->length);
        eval->activations[i] = create_vector(nn->biases[i]->length);
    }

    return eval;
}

void delete_evaluator(Evaluator *eval) {

    for (int i = 0; i < eval->layers; i++) {
        delete_vector(eval->neurons[i]);
        delete_vector(eval->activations[i]);
    }

    free(eval->neurons);
    free(eval->activations);
    free(eval);
}

void print_evaluator(Evaluator *eval) {

    printf("==========================================\n");
    printf("======== Cached Evalution Nuerons ========\n\n");

    for (int i = 0; i < eval->layers; i++) {
        printf("Layer %d Neurons:\n", i + 1);
        print_vector(eval->neurons[i]);
        printf("Layer %d Activations:\n", i + 1);
        print_vector(eval->activations[i]);
    }

    printf("==========================================\n");
}


void activate_null(Vector *input, Vector *output) {

    assert(input->length == output->length);

    for (int i = 0; i < input->length; i++)
        output->values[i] = input->values[i];
}

void activate_relu(Vector *input, Vector *output) {

    assert(input->length == output->length);

    for (int i = 0; i < input->length; i++)
        output->values[i] = relu(input->values[i]);
}

void activate_sigmoid(Vector *input, Vector *output) {

    assert(input->length == output->length);

    for (int i = 0; i < input->length; i++)
        output->values[i] = sigmoid(input->values[i]);
}


void dense_evaluate_network(Network *nn, Evaluator *eval, Vector *input) {

    assert(nn->layers == eval->layers);

    int layer = 0;

    // Input Layer

    {
        affine_transform_bias(input, nn->weights[0], nn->biases[0], eval->neurons[0]);
        activate_relu(eval->neurons[0], eval->activations[0]);
        layer++;
    }

    // Hidden Layers

    while (layer < nn->layers - 1) {

        affine_transform_bias(
            eval->activations[layer-1], nn->weights[layer],
            nn->biases[layer], eval->neurons[layer]
        );

        activate_relu(eval->neurons[layer], eval->activations[layer]);
        layer++;
    }

    // Output Layer

    {
        affine_transform_bias(
            eval->activations[layer-1], nn->weights[layer],
            nn->biases[layer], eval->neurons[layer]
        );

        activate_sigmoid(eval->neurons[layer], eval->activations[layer]);
    }
}

/**************************************************************************************************************/

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-SIGM_COEFF * x));
}

float sigmoid_prime(float x) {
    float sigm = sigmoid(x);
    return SIGM_COEFF * sigm * (1.0 - sigm);
}

float loss_function(float x, float y) {
    return pow(y - x, 2.0);
}

float loss_prime(float x, float y) {
    return -2.0 * (y - x);
}

float relu(float x) {
    return fmaxf(0.0, x);
}

float relu_prime(float x) {
    return x > 0.0 ? 1.0 : 0.0;
}

/**************************************************************************************************************/

Gradient *create_gradient(Network *nn) {

    Gradient *grad = malloc(sizeof(Gradient));

    grad->layers  = nn->layers;
    grad->weights = malloc(sizeof(Matrix*) * grad->layers);
    grad->biases  = malloc(sizeof(Vector*) * grad->layers);

    for (int i = 0; i < grad->layers; i++) {
        grad->weights[i] = create_matrix(nn->weights[i]->rows, nn->weights[i]->cols);
        grad->biases[i]  = create_vector(nn->biases[i]->length);
    }

    return grad;
}

void delete_gradient(Gradient *grad) {
    delete_network(grad);
}

void print_gradient(Gradient *grad) {
    print_network(grad);
}

void zero_gradient(Gradient *grad) {

    for (int i = 0; i < grad->layers; i++) {

        for (int j = 0; j < grad->weights[i]->rows * grad->weights[i]->cols; j++)
            grad->weights[i]->values[j] = 0.0;

        for (int j = 0; j < grad->biases[i]->length; j++)
            grad->biases[i]->values[j] = 0.0;
    }
}


void build_backprop_grad(Network *nn, Evaluator *eval, Gradient *grad, Vector *sample, float result) {

    const int layers = nn->layers;
    const int final  = nn->layers - 1;

    float loss = loss_function(eval->activations[final]->values[0], result);
    float dloss_dout = loss_prime(eval->activations[final]->values[0], result);

    float delta[] = { dloss_dout };

    apply_backprop(nn, eval, grad, sample, delta, final);

    print_gradient(grad);
}

void apply_backprop(Network *nn, Evaluator *eval, Gradient *grad, Vector *sample, float *delta, int layer) {

    const int layers = nn->layers;
    const int final  = nn->layers - 1;

    if (layer == 0)
        return apply_backprop_input(nn, eval, grad, sample, delta);

    float (*activation_prime)(float) = layer == final ? &sigmoid_prime : &relu_prime;

    for (int i = 0; i < eval->neurons[layer]->length; i++)
        delta[i] *= activation_prime(eval->neurons[layer]->values[i]);

    add_vector(grad->biases[layer], delta);
    add_matrix_vec_mul_vec(grad->weights[layer], delta, eval->activations[layer-1]);

    float delta_d1[grad->weights[layer]->rows];

    for (int i = 0; i < grad->weights[layer]->rows; i++) {
        delta_d1[i] = 0.0;
        for (int j = 0; j < grad->weights[layer]->cols; j++)
            delta_d1[i] += delta[j] * nn->weights[layer]->values[i * grad->weights[layer]->cols + j];
    }

    apply_backprop(nn, eval, grad, sample, delta_d1, layer-1);
}

void apply_backprop_input(Network *nn, Evaluator *eval, Gradient *grad, Vector *sample, float *delta) {

    const int layers = nn->layers;
    const int final  = nn->layers - 1;
    const int layer  = 0;

    float (*activation_prime)(float) = layer == final ? &sigmoid_prime : &relu_prime;

    for (int i = 0; i < eval->neurons[layer]->length; i++)
        delta[i] *= activation_prime(eval->neurons[layer]->values[i]);

    add_vector(grad->biases[layer], delta);
    add_matrix_vec_mul_vec(grad->weights[layer], delta, sample);
}

/**************************************************************************************************************/