#include "config.h"
#include "mybmp.h"
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

void matmul(float *A, float *B, float *C, int n) { // only for square matrices
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        C[k * n + i] += B[j * n + i] * A[k * n + j];
      }
    }
  }
}
void matmul_tester() {
  float A[4] = {1, 2, 3, 4};
  float B[4] = {-4, -3, -2, -1};
  float C[4] = {0, 0, 0, 0};
  int n = 2;
  matmul(A, B, C, n);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      printf("%.2f ", C[i * n + j]);
    }
    printf("\n");
  }
  if (C[0] == -8.0 && C[1] == -5.0 && C[2] == -20.0 && C[3] == -13.0)
    printf("CPU matmul correct\n");
  else
    printf("!ERROR IN CPU MATMUL!\n");
  printf("=========");
}

void lilfbig(uint32_t *big) {
  *big = (*big >> 24) | (*big << 24) | ((*big & 0x0000FF00) << 8) |
         ((*big & 0x00FF0000) >> 8);
}

int main() {
  matmul_tester();

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
    // lilfbig(train_data);
  }

  // for (int i = 0; i < 28; i++) {
  //   printf("%d\n", train_data[0 + 12 * sizes[1] + i]);
  // }

  uint8_t sample[sizes[1] * sizes[2]];
  memcpy(sample, train_data, sizes[1] * sizes[2] * sizeof(uint8_t));
  draw((uint8_t)28, (uint8_t)28, (uint8_t)1, sample);

  return 0;
}
