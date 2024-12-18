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
  float *dW;
  float *dH;
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
  layer.dW = NULL;
  layer.dH = NULL;
  return layer;
}
Layer create_relu(int size) {
  Layer layer;
  layer.type = RELU;
  layer.ins = size;
  layer.outs = size;
  layer.W = NULL;
  layer.H = NULL;
  layer.dW = NULL;
  layer.dH = NULL;
  return layer;
}
Layer create_softmax(int size) {
  Layer layer;
  layer.type = SOFTMAX;
  layer.ins = size;
  layer.outs = size;
  layer.W = NULL;
  layer.H = NULL;
  layer.dW = NULL;
  layer.dH = NULL;
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
    layer->H = (float *)malloc(sizeof(float) * input_size * layer->ins);
    for (int i = 0; i < input_size * layer->ins; i++) {
      layer->H[i] = input[i] > 0 ? input[i] : 0;
    }
    break;
  }
  case SOFTMAX: {
    layer->H = (float *)malloc(sizeof(float) * input_size * layer->ins);
    for (int i = 0; i < input_size; i++) { // for each sample
      // Find max for numerical stability
      float max_val = input[0 * input_size + i];
      for (int j = 1; j < layer->ins; j++) {
        if (input[j * input_size + i] > max_val) {
          max_val = input[j * input_size + i];
        }
      }

      // Compute sum of exp(x - max_val)
      float sum = 0.0f;
      for (int j = 0; j < layer->ins; j++) {
        sum += exp(input[j * input_size + i] - max_val);
      }

      // Compute softmax
      for (int j = 0; j < layer->ins; j++) {
        layer->H[j * input_size + i] =
            exp(input[j * input_size + i] - max_val) / sum;
      }
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
    current_input = layer_forward(&net->layers[i], current_input, input_size);
  }
  return current_input;
}

float *layer_backward(Layer *layer, float *dH_above, int *size_above) {
  if (layer->dH != NULL)
    free(layer->dH);
  if (layer->dW != NULL)
    free(layer->dW);

  switch (layer->type) {
  case DENSE: {
    break;
  }
  case RELU: {
    layer->dH = (float *)malloc(sizeof(float) * size_above[0] * size_above[1]);
    for (int i = 0; i < size_above[0] * size_above[1]; i++) {
        layer->dH[i] = layer->H[i] > 0 ? dH_above[i] : 0;
    }
    break;
  }
  case SOFTMAX: {
    layer->dH = (float *)malloc(sizeof(float) * size_above[0] * size_above[1]);
    // 10x60000 -> 10x60000
    for (int sample = 0; sample < size_above[1]; sample++) {
      for (int i = 0; i < size_above[0]; i++) { // loop over input's 10 classes
        float sum = 0.0f;
        for (int j = 0; j < size_above[0]; j++) { // loop over output's 10 classes
          float yi = layer->H[i * size_above[1] + sample];
          float yj = layer->H[j * size_above[1] + sample];
          float grad_j = dH_above[j * size_above[1] + sample];

          if (i == j) { // when we're dealing with calculating how the output for the j th class gets effected by change in that class in the input
            sum += grad_j * yi * (1.0 -yi);
          }
          else {
            sum += grad_j * (-yi * yj);
          }
        }
        layer->dH[i * size_above[1] + sample] = sum;
      }
    }
    break;
  }
  }
  return layer->dH;
}

void backward(Network *net, float loss, int *sizes, int *y) {
  float *previous_output = net->layers[net->num_layers - 1].H;
  float *grad = (float *)malloc(sizeof(float) * sizes[0] * sizes[1]);

  for (int i = 0; i < sizes[1]; i++) {
    for (int j = 0; j < sizes[0]; j++) {
      if (j == y[i]) {
        grad[j * sizes[1] + i] =
            (-1.0f / previous_output[j * sizes[1] + i]) / sizes[1];
      } else {
        grad[j * sizes[1] + i] = 0.0f;
      }
    }
  }
  int size_above[2] = {sizes[0], sizes[1]};
  for (int i = net->num_layers; i > 0; i--) {
    layer_backward(&net->layers[i - 1], grad, size_above);
    size_above[0] = net->layers[i - 1].ins;
  }
}

int *get_prediction(float *h, int *sizes) {
  int *predictions = (int *)malloc(sizeof(int) * sizes[0] * sizes[1]);
  for (int i = 0; i < sizes[1]; i++) {
    float pred = 0;
    int max;
    for (int j = 0; j < sizes[0]; j++) {
      if (h[j * sizes[1] + i] > pred) {
        pred = h[j * sizes[1] + i];
        max = j;
      }
    }
    predictions[i] = max;
  }
  return predictions;
}

float calc_loss(float *h, int *y, int *sizes) {
  float loss = 0.0f;
  const float epsilon = 1e-7f;

  for (int i = 0; i < sizes[1]; i++) {
    float true_class_prob = h[y[i] * sizes[1] + i];
    true_class_prob = fmaxf(true_class_prob, epsilon); // return max of two
    loss -= logf(true_class_prob);
  }
  loss /= sizes[1];
  return loss;
}

int main() {
  FILE *train_images = fopen(DATA_DIR "train-images-idx3-ubyte", "rb");
  FILE *train_labels = fopen(DATA_DIR "train-labels-idx1-ubyte", "rb");
  // FILE *test_images = fopen(DATA_DIR "t10k-images-idx3-ubyte", "rb");
  // FILE *test_labels = fopen(DATA_DIR "t10k-labels-idx3-ubyte", "rb");

  // read first 32 bits / 4 bytes
  uint32_t magic_number;
  fread(&magic_number, sizeof(uint32_t), 1, train_images);
  // big endian: first byte of the 4 byte magic number is on the right
  // | 0x03 | 0x08 | 00000000 | *00000000* |
  // printf("magic number: %d\n", magic_number);
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
  fclose(train_images);

  fseek(train_labels, 8, SEEK_SET);
  int *train_classes = (int *)malloc(sizeof(int) * 60000);
  uint8_t label;
  int la = 0;
  for (int i = 0; i < 60000; i++) {
    fread(&label, 1, 1, train_labels);
    train_classes[i] = (int)label;
  }
  fclose(train_labels);

  int train_size[2] = {data_size[0], data_size[1] * data_size[2]};
  float *input;
  input = transpose(train_data, train_size);

  Network net = create_network();
  add_layer(&net, create_dense(train_size[1], 128));
  add_layer(&net, create_relu(128));
  add_layer(&net, create_dense(128, 64));
  add_layer(&net, create_relu(64));
  add_layer(&net, create_dense(64, 10));
  add_layer(&net, create_softmax(10));

  int out_size[2] = {10, train_size[0]};
  int iters = 2;
  for (int i = 0; i < iters; i++) {
    float *h = forward(&net, input, data_size[0]);
    float loss = calc_loss(h, train_classes, out_size);
    printf("iter=%d -- loss=%.4f", i, loss);
    backward(&net, loss, out_size, train_classes);
  }

  free_network(&net);
  return 0;
}
