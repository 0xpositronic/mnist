#include "config.h"
#include "mybmp.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

void print_mat(float *A, int *sizes) {
  for (int i = 0; i < sizes[0]; i++) {
    for (int j = 0; j < sizes[1]; j++) {
      printf("%.2f ", A[i * sizes[1] + j]);
    }
    printf("\n");
  }
}

void matmul(float *A, float *B, float *C, int *sizes) { // less slow
  for (int i = 0; i < sizes[3]; i++) {
    for (int j = 0; j < sizes[0]; j++) {
      float sum = 0;
      for (int k = 0; k < sizes[2]; k++) {
        sum += A[j * sizes[1] + k] * B[k * sizes[3] + i];
      }
      C[j * sizes[3] + i] = sum;
    }
  }
}
void matmul_tester() {
  float A[6] = {1, 2, 3, 4, 5, 6};
  float B[3] = {-6, -5, -4};
  float C[2] = {0, 0};
  int sizes[4] = {2, 3, 3, 1};
  int out_sizes[2] = {sizes[0], sizes[3]};
  matmul(A, B, C, sizes);
  print_mat(C, out_sizes);
}

float *transpose(float *A, int *sizes) {
  float *At = (float *)malloc(sizeof(float) * sizes[0] * sizes[1]);
  for (int i = 0; i < sizes[0]; i++) {
    for (int j = 0; j < sizes[1]; j++) {
      At[j * sizes[0] + i] = A[i * sizes[1] + j];
    }
  }
  return At;
}
void transpose_tester() {
  float A[6] = {1, 2, 3, 4, 5, 6};
  int sizes[2] = {2, 3};
  float *At = transpose(A, sizes);
  int sizest[2] = {sizes[1], sizes[0]};
  print_mat(A, sizes);
  print_mat(At, sizest);
}

float *relu(float *A, int size) {
  float *Ar = (float *)malloc(sizeof(float) * size);
  for (int i = 0; i < size; i++) {
    if (A[i] < 0) {
      Ar[i] = 0;
    } else
      Ar[i] = A[i];
  }
  return Ar;
}

float *softmax(float *A, int size) {
  float *As = (float *)malloc(sizeof(float) * size);
  float sum = 0.0;
  for (int i = 0; i < size; i++) {
    sum += exp(A[i]);
  }
  for (int i = 0; i < size; i++) {
    As[i] = exp(A[i]) / sum;
  }
  return As;
}

float *zeros(int size) {
  float *A = (float *)malloc(sizeof(float) * size);
  for (int i = 0; i < size; i++) {
    A[i] = 0;
  }
  return A;
}
float *fill(int size) {
  float *A = (float *)malloc(sizeof(float) * size);
  for (int i = 0; i < size; i++) {
    A[i] = i;
  }
  return A;
}

void lilfbig(uint32_t *big) { // big-endian to little-endian
  *big = (*big >> 24) | (*big << 24) | ((*big & 0x0000FF00) << 8) |
         ((*big & 0x00FF0000) >> 8);
}

void save_matrix(float *data, int rows, int cols, const char *filename) {
  FILE *f = fopen(filename, "wb");
  fwrite(&rows, sizeof(int), 1, f);
  fwrite(&cols, sizeof(int), 1, f);
  fwrite(data, sizeof(float), rows * cols, f);
  fclose(f);
}

float *xavier_init(int fan_in, int fan_out) {
  float *weights = (float *)malloc(sizeof(float) * fan_in * fan_out);
  float scale = sqrtf(2.0f / (fan_in + fan_out));

  srand(1);

  for (int i = 0; i < fan_in * fan_out; i++) {
    // Generate random number between -1 and 1
    float rand_val = ((float)rand() / RAND_MAX) * 2 - 1;
    weights[i] = rand_val * scale;
  }
  return weights;
}

float *forward(float *weights1, float *weights2, float *weights3, float *out1,
               float *out2, float *out3, int *layer_sizes1, int *layer_sizes2,
               int *layer_sizes3, int *data_size, float *train_data_T) {
  float *h1, *h2, *h3;
  matmul(weights1, train_data_T, out1, layer_sizes1);
  h1 = relu(out1, 128 * data_size[0]);
  matmul(weights2, h1, out2, layer_sizes2);
  h2 = relu(out2, 64 * data_size[0]);
  matmul(weights3, h2, out3, layer_sizes3);
  h3 = softmax(out3, 10 * data_size[0]);
  return h3;
}

int main() {
  matmul_tester();
  transpose_tester();

  FILE *train_images = fopen(DATA_DIR "train-images-idx3-ubyte", "rb");
  FILE *train_labels = fopen(DATA_DIR "train-labels-idx1-ubyte", "rb");
  FILE *test_images = fopen(DATA_DIR "t10k-images-idx3-ubyte", "rb");
  FILE *test_labels = fopen(DATA_DIR "t10k-labels-idx3-ubyte", "rb");

  // read first 32 bits / 4 bytes
  uint32_t magic_number;
  fread(&magic_number, sizeof(uint32_t), 1, train_images);

  // big endian: first byte of the 4 byte magic number is on the right
  // | 0x03 | 0x08 | 00000000 | *00000000* |
  printf("magic number: %d\n", magic_number);
  // 50855936 = 11000010000000000000000000
  /*
  Little endian
  Address	0	1	2	3
  Data	00	00	08	03

  Big endian
  Address	0	1	2	3
  Data	03	08	00	00
  */

  // get the 3rd and 4th bytes. 0xFF is 11111111, a full byte.
  uint8_t data_type = (magic_number >> 16) & 0xFF;
  uint8_t dimension = (magic_number >> 24) & 0xFF;
  printf("data type: %d\n", data_type);
  printf("dimensions: %d\n", dimension);

  int train_size[dimension];
  uint32_t size;
  size_t total_size = 1;
  for (int i = 0; i < dimension; i++) {
    fread(&size, sizeof(uint32_t), 1, train_images);
    lilfbig(&size);
    total_size *= size;
    train_size[i] = (int)size;
    printf("%d\n", size);
  }
  printf("total data size: %lu\n", total_size);

  float *train_data = (float *)malloc(sizeof(float) * total_size);
  uint8_t pixel;
  for (int i = 0; i < total_size; i++) {
    fread(&pixel, sizeof(uint8_t), 1, train_images);
    train_data[i] = ((float)pixel / 127.5f) - 1.0f;
  }

  float *weights1 = xavier_init(128, 28 * 28);
  float *weights2 = xavier_init(64, 128);
  float *weights3 = xavier_init(10, 64);

  float *out1 = zeros(train_size[0] * 128);
  float *out2 = zeros(train_size[0] * 64);
  float *out3 = zeros(train_size[0] * 10);

  int data_size[2] = {train_size[0], train_size[1] * train_size[2]};
  float *train_data_T = transpose(train_data, data_size);

  int layer_sizes1[4] = {128, 28 * 28, data_size[1], data_size[0]};
  int layer_sizes2[4] = {64, 128, 128, data_size[0]};
  int layer_sizes3[4] = {10, 64, 64, data_size[0]};

  for (int iter = 0; iter < 10; iter++) {
    float *output =
        forward(weights1, weights2, weights3, out1, out2, out3, layer_sizes1,
                layer_sizes2, layer_sizes3, data_size, train_data_T);
    // calculate loss. loss()
    // update weights. backward()
  }
  // calculate final loss and accuracy
  // save weights
  return 0;
}
