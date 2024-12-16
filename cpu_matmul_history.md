void matmul(float *A, float *B, float *C, int *sizes) { // slow
  for (int i = 0; i < sizes[3]; i++) {
    for (int j = 0; j < sizes[2]; j++) {
      for (int k = 0; k < sizes[0]; k++) {
        C[k * sizes[3] + i] += B[j * sizes[3] + i] * A[k * sizes[1] + j];
      }
    }
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

