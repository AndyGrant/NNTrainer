#pragma once

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/**************************************************************************************************************/

typedef struct Vector {
    int length;
    float *values;
} Vector;

typedef struct Matrix {
    int rows, cols;
    float *values;
} Matrix;

#define PRINT_FORMAT "%+10.6f"

Vector *create_vector(int length);
void delete_vector(Vector *vec);
void print_vector(const Vector *vec);
void set_vector(Vector *vec, int length, float *values);
void zero_vector(Vector *vec);

Matrix *create_matrix(int rows, int cols);
void delete_matrix(Matrix *mat);
void print_matrix(const Matrix *mat);
void set_matrix_row(Matrix *mat, int row, int cols, float *values);
void set_matrix_col(Matrix *mat, int col, int rows, float *values);

void add_vector(Vector *vec, float *addends);
void add_matrix_vec_mul_vec(Matrix *mat, float *vec1, Vector *vec2);
void set_vector_vec_mul_mat(float *output, float *vec, Matrix *mat);
void mul_vector_func_of_vec(float *delta, Vector *vec, float (*func)(float));

void affine_transform(Vector *vector, Matrix *matrix, Vector *output);
void affine_transform_bias(Vector *vector, Matrix *matrix, Vector *bias, Vector *output);

/**************************************************************************************************************/

typedef struct Network {
    Matrix **weights;
    Vector **biases;
    int layers;
} Network;

Network *create_network(int layers, ...);
void delete_network(Network *nn);
void print_network(Network *nn);
void randomize_network(Network *nn);

typedef struct Evaluator {
    Vector **neurons;
    Vector **activations;
    int layers;
} Evaluator;

Evaluator *create_evaluator(Network *nn);
void delete_evaluator(Evaluator *eval);
void print_evaluator(Evaluator *eval);

void activate_null(Vector *input, Vector *output);
void activate_relu(Vector *input, Vector *output);
void activate_sigmoid(Vector *input, Vector *output);

void dense_evaluate_network(Network *nn, Evaluator *eval, Vector *input);

/**************************************************************************************************************/

#define SIGM_COEFF (3.145 / 400.000)

float sigmoid(float x);
float sigmoid_prime(float x);
float loss_function(float x, float y);
float loss_prime(float x, float y);
float relu(float x);
float relu_prime(float x);

/**************************************************************************************************************/

typedef Network Gradient;

Gradient *create_gradient(Network *nn);
void delete_gradient(Gradient *grad);
void print_gradient(Gradient *grad);
void zero_gradient(Gradient *grad);

void build_backprop_grad(Network *nn, Evaluator *eval, Gradient *grad, Vector *sample, float result);
void apply_backprop(Network *nn, Evaluator *eval, Gradient *grad, Vector *sample, float *delta, int layer);
void apply_backprop_input(Network *nn, Evaluator *eval, Gradient *grad, Vector *sample, float *delta);

/**************************************************************************************************************/

#define MAX_INDICIES 32
#define NSAMPLES 16384
#define DATAFILE "halogen.data"

typedef struct Sample {
    float result;
    int length, indices[MAX_INDICIES];
} Sample;

Sample *load_samples(char *fname, int length);
void load_sample(FILE *fin, Sample *sample);
void vectorify_sample(Vector *vec, Sample *sample);

/**************************************************************************************************************/

void update_weights(Network *nn, Gradient *grad, float lrate, int batch_size);

/**************************************************************************************************************/