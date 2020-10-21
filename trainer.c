#include <assert.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "trainer.h"

int main() {

    Sample *samples = load_samples(DATAFILE, NSAMPLES);
    Network *nn     = create_network(2, 768, 96, 1);
    Optimizer *opt  = create_optimizer(nn);
    Evaluator *eval = create_evaluator(nn);
    Gradient *grad  = create_gradient(nn);

    for (int epoch = 0; epoch < 25000; epoch++) {

        float loss = 0.0;

        for (int i = 0; i < NSAMPLES; i++) {

            sparse_evaluate_network(nn, eval, &samples[i]);
            build_backprop_grad(nn, eval, grad, &samples[i]);

            loss += loss_function(eval->activations[nn->layers-1]->values[0], samples[i].result);

            if ((i && i % BATCHSIZE == 0))
                update_network(opt, nn, grad, LEARNRATE, BATCHSIZE);
        }

        // update_network(opt, nn, grad, LEARNRATE, NSAMPLES);

        printf("[Epoch %5d] Loss = %.9f\n", epoch, loss / NSAMPLES);
        fflush(stdout);

        char fname[512];
        sprintf(fname, "%sepoch%d.nn", "Networks/", epoch);
        save_network(nn, fname);
    }
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


void input_transform(Sample *sample, Matrix *matrix, Vector *bias, Vector *output) {

    assert(output->length == bias->length);
    assert(output->length == matrix->cols);

    for (int i = 0; i < output->length; i++)
        output->values[i] = bias->values[i];

    for (int i = 0; i < sample->length; i++)
        for (int j = 0; j < matrix->cols; j++)
            output->values[j] += matrix->values[sample->indices[i] * matrix->cols + j];
}

void affine_transform(Vector *vector, Matrix *matrix, Vector *bias, Vector *output) {

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

void set_vector_vec_mul_mat(float *output, float *vec, Matrix *mat) {

    for (int i = 0; i < mat->rows; i++) {
        output[i] = 0.0;
        for (int j = 0; j < mat->cols; j++)
            output[i] += vec[j] * mat->values[i * mat->cols + j];
    }
}

void mul_vector_func_of_vec(float *delta, Vector *vec, float (*func)(float)) {

    for (int i = 0; i < vec->length; i++)
        delta[i] *= func(vec->values[i]);
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

    #define random_weight() (((rand() % 10000) - 5000) / 5000.0)

    for (int i = 0; i < nn->layers; i++) {

        for (int j = 0; j < nn->weights[i]->rows * nn->weights[i]->cols; j++)
            nn->weights[i]->values[j] = random_weight();

        for (int j = 0; j < nn->biases[i]->length; j++)
            nn->biases[i]->values[j] = random_weight();
    }

    #undef random_weight
}


void save_network(Network *nn, char *fname) {

    FILE *fout = fopen(fname, "w");

    for (int layer = 0; layer < nn->layers; layer++) {

        const int rows = nn->weights[layer]->rows;
        const int cols = nn->weights[layer]->cols;

        for (int col = 0; col < cols; col++) {
            fprintf(fout, "\"%d ", rows);
            for (int row = 0; row < rows; row++)
                fprintf(fout, "%f ", nn->weights[layer]->values[row * cols + col]);
            fprintf(fout, "%f\",\n", nn->biases[layer]->values[col]);
        }
    }

    fclose(fout);
}

void load_network(Network *nn, char *fname) {
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


void sparse_evaluate_network(Network *nn, Evaluator *eval, Sample *sample) {

    assert(nn->layers == eval->layers);

    int layer = 0;

    // Input Layer

    {
        input_transform(sample, nn->weights[0], nn->biases[0], eval->neurons[0]);
        activate_relu(eval->neurons[0], eval->activations[0]);
        layer++;
    }

    // Hidden Layers

    while (layer < nn->layers - 1) {

        affine_transform(
            eval->activations[layer-1], nn->weights[layer],
            nn->biases[layer], eval->neurons[layer]
        );

        activate_relu(eval->neurons[layer], eval->activations[layer]);
        layer++;
    }

    // Output Layer

    {
        affine_transform(
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


void build_backprop_grad(Network *nn, Evaluator *eval, Gradient *grad, Sample *sample) {

    const int final  = nn->layers - 1;

    float dloss_dout = loss_prime(eval->activations[final]->values[0], sample->result);

    float delta[] = { dloss_dout };

    apply_backprop(nn, eval, grad, sample, delta, final);
}

void apply_backprop(Network *nn, Evaluator *eval, Gradient *grad, Sample *sample, float *delta, int layer) {

    const int final  = nn->layers - 1;

    if (layer == 0) return apply_backprop_input(nn, eval, grad, sample, delta);

    float delta_d1[grad->weights[layer]->rows];
    float (*activation_prime)(float) = layer == final ? &sigmoid_prime : &relu_prime;

    mul_vector_func_of_vec(delta, eval->neurons[layer], activation_prime);
    add_vector(grad->biases[layer], delta);
    add_matrix_vec_mul_vec(grad->weights[layer], delta, eval->activations[layer-1]);
    set_vector_vec_mul_mat(delta_d1, delta, nn->weights[layer]);
    apply_backprop(nn, eval, grad, sample, delta_d1, layer-1);
}

void apply_backprop_input(Network *nn, Evaluator *eval, Gradient *grad, Sample *sample, float *delta) {

    (void) nn;

    mul_vector_func_of_vec(delta, eval->neurons[0], &relu_prime);
    add_vector(grad->biases[0], delta);

    for (int i = 0; i < sample->length; i++)
        for (int j = 0; j < grad->weights[0]->cols; j++)
            grad->weights[0]->values[sample->indices[i] * grad->weights[0]->cols + j] += delta[j];

    // add_matrix_vec_mul_vec(grad->weights[0], delta, sample);
}

/**************************************************************************************************************/

Sample *load_samples(char *fname, int length) {

    Sample *samples = malloc(sizeof(Sample) * length);
    printf("Allocated %.2fMB for Samples\n",
        (float)(sizeof(Sample) * length) / (1024 * 1024));

    FILE *fin = fopen(fname, "r");

    for (int i = 0; i < length; i++) {

        load_sample(fin, &samples[i]);

        if (i == length - 1 || i % (1024*256) == 0) {
            printf("\rLoaded %d of %d Samples", i+1, length);
            fflush(stdout);
        }
    }

    printf("\nFinished Reading %s\n\n", fname);

    fclose(fin);

    return samples;
}

void load_sample(FILE *fin, Sample *sample) {

    char *ptr, line[1024];
    if (fgets(line, 1024, fin) == NULL)
        exit(EXIT_FAILURE);

    sample->length = 0;
    sample->result = atof(strtok(line, " "));

    while ((ptr = strtok(NULL, " ")) != NULL)
        sample->indices[sample->length++] = atoi(ptr);
}

/**************************************************************************************************************/

Optimizer *create_optimizer(Network *nn) {
    Optimizer *opt = malloc(sizeof(Optimizer));
    opt->momentum = create_gradient(nn);
    opt->velocity = create_gradient(nn);
    return opt;
}

void delete_optimizer(Optimizer *opt) {
    free(opt->momentum);
    free(opt->velocity);
    free(opt);
}

void update_network(Optimizer *opt, Network *nn, Gradient *grad, float lrate, int batch_size) {

    for (int layer = 0; layer < nn->layers; layer++) {

        const int rows = nn->weights[layer]->rows;
        const int cols = nn->weights[layer]->cols;

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < rows * cols; i++) {

            opt->momentum->weights[layer]->values[i]
                = (BETA_1 * opt->momentum->weights[layer]->values[i])
                + (1 - BETA_1) * (grad->weights[layer]->values[i] / batch_size);

            opt->velocity->weights[layer]->values[i]
                = (BETA_2 * opt->velocity->weights[layer]->values[i])
                + (1 - BETA_2) * pow(grad->weights[layer]->values[i] / batch_size, 2.0);

            nn->weights[layer]->values[i] -= lrate * opt->momentum->weights[layer]->values[i]
                                           * (1.0 / (1e-8 + sqrt(opt->velocity->weights[layer]->values[i])));
        }

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < nn->biases[layer]->length; i++) {

            opt->momentum->biases[layer]->values[i]
                = (BETA_1 * opt->momentum->biases[layer]->values[i])
                + (1 - BETA_1) * (grad->biases[layer]->values[i] / batch_size);

            opt->velocity->biases[layer]->values[i]
                = (BETA_2 * opt->velocity->biases[layer]->values[i])
                + (1 - BETA_2) * pow(grad->biases[layer]->values[i] / batch_size, 2.0);

            nn->biases[layer]->values[i] -= lrate * opt->momentum->biases[layer]->values[i]
                                          * (1.0 / (1e-8 + sqrt(opt->velocity->biases[layer]->values[i])));
        }
    }

    zero_gradient(grad);
}

/**************************************************************************************************************/
