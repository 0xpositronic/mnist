#include "config.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

typedef enum { DENSE, RELU, SOFTMAX } LayerType;
typedef struct {
  LayerType type;
  int ins;
  int outs;
  float *W;
  float *H;
} Layer;
typedef struct {
  Layer *layers;
  int num_layers;
} Network;

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

float *transpose(float *A, int *sizes) {
  float *At = (float *)malloc(sizeof(float) * sizes[0] * sizes[1]);
  for (int i = 0; i < sizes[0]; i++) {
    for (int j = 0; j < sizes[1]; j++) {
      At[j * sizes[0] + i] = A[i * sizes[1] + j];
    }
  }
  return At;
}

float *xavier_init(int ins, int outs) {
  float *weights = (float *)malloc(sizeof(float) * ins * outs);
  float scale = sqrtf(2.0f / (ins + outs));

  srand(1);

  for (int i = 0; i < ins * outs; i++) {
    // Generate random number between -1 and 1
    float rand_val = ((float)rand() / RAND_MAX) * 2 - 1;
    weights[i] = rand_val * scale;
  }
  return weights;
}

Layer create_dense(int ins, int outs) {
  Layer layer;
  layer.type = DENSE;
  layer.ins = ins;
  layer.outs = outs;
  layer.W = xavier_init(outs, ins);
  layer.H = NULL; // cant know the batch size here
  return layer;
}
Layer create_relu() {
  Layer layer;
  layer.type = RELU;
  layer.ins = 0;
  layer.outs = 0;
  layer.W = NULL;
  layer.H = NULL;
  return layer;
}
Layer create_softmax() {
  Layer layer;
  layer.type = SOFTMAX;
  layer.ins = 0;
  layer.outs = 0;
  layer.W = NULL;
  layer.H = NULL;
  return layer;
}

void free_layer(Layer *layer) {
  if (layer->W != NULL)
    free(layer->W);
  if (layer->H != NULL)
    free(layer->H);
}

float *layer_forward(Layer *layer, float *input, int input_size) {
  if (layer->H != NULL)
    free(layer->H); // free previous output since we will create a new one, we
                    // would lose the previous.

  switch (layer->type) {
  case DENSE: {
    layer->H = (float *)malloc(
        sizeof(float) * layer->outs *
        input_size); // no need to init to 0 since we use mid sum

    int sizes[4] = {layer->outs, layer->ins, layer->ins, input_size};
    matmul(layer->W, input, layer->H, sizes);
    break;
  }
  case RELU: {
    layer->ins = input_size;
    layer->outs = input_size;
    layer->H = (float *)malloc(sizeof(float) * input_size);
    for (int i = 0; i < input_size; i++) {
      layer->H[i] = input[i] > 0 ? input[i] : 0;
    }
    break;
  }
  case SOFTMAX: {
    layer->ins = input_size;
    layer->outs = input_size;
    layer->H = (float *)malloc(sizeof(float) * input_size);
    float sum = 0.0f;
    for (int i = 0; i < input_size; i++) {
      sum += exp(input[i]);
    }
    for (int i = 0; i < input_size; i++) {
      layer->H[i] = exp(input[i]) / sum;
    }
    break;
  }
  }
  return layer->H;
}

Network create_network() {
  Network net;
  net.layers = NULL;
  net.num_layers = 0;
  return net;
}
void free_network(Network *net) {
  for (int i = 0; i < net->num_layers; i++) {
    free_layer(&net->layers[i]);
  }
  free(net->layers);
}
void add_layer(Network *net, Layer layer) {
  net->num_layers++;
  net->layers = (Layer *)realloc(net->layers, sizeof(Layer) * net->num_layers);
  net->layers[net->num_layers - 1] = layer;
}
float *forward(Network *net, float *input, int input_size) {
  float *current_input = input;
  for (int i = 0; i < net->num_layers; i++) {
    printf("Layer %d forward pass (type=%d, ins=%d, outs=%d, input_size=%d)\n",
           i, net->layers[i].type, net->layers[i].ins, net->layers[i].outs,
           input_size);
    current_input = layer_forward(&net->layers[i], current_input, input_size);
  }
  return current_input;
}

int main() {
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

  int data_size[dimension];
  uint32_t size;
  size_t total_size = 1;
  for (int i = 0; i < dimension; i++) {
    fread(&size, sizeof(uint32_t), 1, train_images);
    lilfbig(&size);
    total_size *= size;
    data_size[i] = (int)size;
    printf("%d\n", size);
  }
  printf("total data size: %lu\n", total_size);

  float *train_data = (float *)malloc(sizeof(float) * total_size);
  uint8_t pixel;
  for (int i = 0; i < total_size; i++) {
    fread(&pixel, sizeof(uint8_t), 1, train_images);
    train_data[i] = ((float)pixel / 127.5f) - 1.0f;
  }

  int train_size[2] = {data_size[0], data_size[1] * data_size[2]};
  float *input;
  input = transpose(train_data, train_size);

  Network net = create_network();
  add_layer(&net, create_dense(train_size[1], 128));
  add_layer(&net, create_relu());
  add_layer(&net, create_dense(128, 64));
  add_layer(&net, create_relu());
  add_layer(&net, create_dense(64, 10));
  add_layer(&net, create_softmax());

  int iters = 10;
  float *h;
  for (int i = 0; i < iters; i++) {
    h = forward(&net, input, data_size[0]);
    // loss = calc_loss(input, labels);
    // net = backward(&net, ...);
  }

  free_network(&net);
  return 0;
}
