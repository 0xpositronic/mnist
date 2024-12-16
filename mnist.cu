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
        C[k * sizes[3] + i] += B[j * sizes[3] + i] * A[k * sizes[1] + j];
      }
    }
  }
}

void mmatmul(float *A, float *B, float *C, int *sizes) {
  // sizes[0] = rows of A
  // sizes[1] = cols of A
  // sizes[2] = cols of B
  // sizes[3] = cols of output

  // Initialize C to zero first
  int output_size = sizes[0] * sizes[3];
  for (int i = 0; i < output_size; i++) {
    C[i] = 0.0f;
  }

  // Matrix multiplication with bounds check
  for (int i = 0; i < sizes[0]; i++) {
    for (int j = 0; j < sizes[3]; j++) {
      float sum = 0.0f;
      for (int k = 0; k < sizes[1]; k++) {
        sum += A[i * sizes[1] + k] * B[k * sizes[3] + j];
      }
      C[i * sizes[3] + j] = sum;
    }
  }
}

void matmul_tester() {
  float A[6] = {1, 2, 3, 4, 5, 6};
  float B[6] = {-6, -5, -4, -3, -2, -1};
  float C[4] = {0, 0, 0, 0};
  int sizes[4] = {2, 3, 3, 2};
  int out_sizes[2] = {sizes[0], sizes[3]};
  matmul(A, B, C, sizes);
  print_mat(C, out_sizes);
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

void lilfbig(uint32_t *big) {
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

#include <float.h>
#include <math.h>

void debug_numerical_issues(float *out, float *weights, float *data,
                            int num_samples) {
  // Check weights
  for (int i = 0; i < 28 * 28 * 1000; i++) {
    if (isnan(weights[i]) || isinf(weights[i])) {
      fprintf(stderr, "Invalid weight at index %d: %f\n", i, weights[i]);
      return;
    }
  }

  // Check input data
  for (int i = 0; i < num_samples * 28 * 28; i++) {
    if (isnan(data[i]) || isinf(data[i])) {
      fprintf(stderr, "Invalid input at index %d: %f\n", i, data[i]);
      return;
    }
  }

  // Check output and get statistics
  float max_val = -FLT_MAX;
  float min_val = FLT_MAX;
  int nan_count = 0;
  int inf_count = 0;
  int large_val_count = 0;

  for (int i = 0; i < num_samples * 1000; i++) {
    if (isnan(out[i])) {
      nan_count++;
      continue;
    }
    if (isinf(out[i])) {
      inf_count++;
      continue;
    }
    if (fabs(out[i]) > 1e6) {
      large_val_count++;
    }
    max_val = fmax(max_val, out[i]);
    min_val = fmin(min_val, out[i]);
  }

  fprintf(stderr, "Output statistics:\n");
  fprintf(stderr, "NaN count: %d\n", nan_count);
  fprintf(stderr, "Inf count: %d\n", inf_count);
  fprintf(stderr, "Large value count: %d\n", large_val_count);
  fprintf(stderr, "Max value: %f\n", max_val);
  fprintf(stderr, "Min value: %f\n", min_val);
}

#include <time.h>
float *xavier_init(int fan_in, int fan_out) {
  float *weights = (float *)malloc(sizeof(float) * fan_in * fan_out);
  float scale = sqrtf(2.0f / (fan_in + fan_out));

  // Use time as seed
  srand(time(NULL));

  for (int i = 0; i < fan_in * fan_out; i++) {
    // Generate random number between -1 and 1
    float rand_val = ((float)rand() / RAND_MAX) * 2 - 1;
    weights[i] = rand_val * scale;
  }
  return weights;
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

  // float *weights = fill(28 * 28 * 1000);
  // float *weights = xavier_init(28 * 28, 1000); // Instead of fill()
  float *weights =
      xavier_init(1000, 28 * 28); // num_output_features x num_input_features
  save_matrix(weights, 1000, 28 * 28, "c_weights.bin");
  float *out = zeros(train_size[0] * 1000);

  int data_size[2] = {train_size[0], train_size[1] * train_size[2]};
  float *train_data_T = transpose(train_data, data_size);

  int layer_sizes[4] = {1000, 28 * 28, data_size[1], data_size[0]};
  matmul(weights, train_data_T, out, layer_sizes);

  debug_numerical_issues(out, weights, train_data, train_size[0]);

  save_matrix(train_data, train_size[0], train_size[1] * train_size[2],
              "c_train_data.bin");
  // save_matrix(weights, 28 * 28, 1000, "c_weights.bin");
  save_matrix(train_data_T, train_size[1] * train_size[2], train_size[0],
              "c_train_data_T.bin");
  save_matrix(out, 1000, train_size[0], "c_output.bin");

  return 0;
}
