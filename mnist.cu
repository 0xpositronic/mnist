#include "config.h"
#include "mybmp.h"
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

void print_mat(float *A, int *sizes) {
  for (int i = 0; i < sizes[0]; i++) {
    for (int j = 0; j < sizes[1]; j++) {
      printf("%.2f ", A[i * sizes[1] + j]);
    }
    printf("\n");
  }
}

void matmul(float *A, float *B, float *C, int *sizes) {
  for (int i = 0; i < sizes[3]; i++) {
    for (int j = 0; j < sizes[2]; j++) {
      for (int k = 0; k < sizes[0]; k++) {
        C[k * sizes[1] + i] += B[j * sizes[3] + i] * A[k * sizes[1] + j];
      }
    }
  }
}
void matmul_tester() {
  float A[4] = {1, 2, 3, 4};
  float B[4] = {-4, -3, -2, -1};
  float C[4] = {0, 0, 0, 0};
  int sizes[4] = {2, 2, 2, 2};
  matmul(A, B, C, sizes);
  print_mat(C, sizes);
  if (C[0] == -8.0 && C[1] == -5.0 && C[2] == -20.0 && C[3] == -13.0)
    printf("CPU matmul correct\n");
  else
    printf("!ERROR IN CPU MATMUL!\n");
}

float *init_weights(int in_size, int out_size) {
  float *weights = (float *)malloc(sizeof(float) * in_size * out_size);
  for (int i = 0; i < in_size * out_size; i++) {
    weights[i] = i;
  }
  return weights;
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
  printf("===\n");
  print_mat(At, sizest);
}

void lilfbig(uint32_t *big) {
  *big = (*big >> 24) | (*big << 24) | ((*big & 0x0000FF00) << 8) |
         ((*big & 0x00FF0000) >> 8);
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

  uint32_t sizes[dimension];
  size_t total_size = 1;
  for (int i = 0; i < dimension; i++) {
    fread(&sizes[i], sizeof(uint32_t), 1, train_images);
    lilfbig(&sizes[i]);
    total_size *= sizes[i];
    printf("%d\n", sizes[i]);
  }
  printf("total data size: %lu\n", total_size);

  uint8_t *train_data = (uint8_t *)malloc(sizeof(uint8_t) * total_size);
  for (int i = 0; i < total_size; i++) {
    fread(&train_data[i], sizeof(uint8_t), 1, train_images);
  }

  uint8_t sample[sizes[1] * sizes[2]];
  memcpy(sample, train_data, sizes[1] * sizes[2] * sizeof(uint8_t));
  draw((uint8_t)28, (uint8_t)28, (uint8_t)1, sample);

  uint8_t data[sizes[1] * sizes[2]];
  memcpy(data, train_data, sizes[1] * sizes[2] * sizeof(uint8_t));
  int image_size[2] = {(int)sizes[1], (int)sizes[2]};

  float *dataf = (float *)malloc(sizeof(float) * total_size);
  for (int i = 0; i < total_size; i++) {
    dataf[i] = ((float)train_data[i] / 127.5f) - 1.0f;
  }
  // print_mat(transpose(dataf, image_size), image_size);

  float *weights = init_weights(28 * 28, 1000);

  int finalsize[2] = {28 * 28, (int)sizes[0]};
  float *datafT = transpose(dataf, finalsize);
  float *out = (float *)malloc(1000 * sizes[0] * sizeof(float));

  for (int i = 0; i < 1000 * sizes[0]; i++) {
    out[i] = 0.0;
  }
  int sizes1[4] = {28 * 28, 1000, finalsize[1], finalsize[0]};
  matmul(weights, datafT, out, sizes1);

  return 0;
}
