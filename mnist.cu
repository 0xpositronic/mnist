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
  float *X;
  float *W;
  float *B;
  float *H;
  float *dW;
  float *dB;
  float *dH;
} Layer;
typedef struct {
  Layer *layers;
  int num_layers;
} Network;

void lilfbig(uint32_t *big) { // big-endian to little-endian
  /*
  big endian: first byte of the 4 byte magic number is on the right
  | 0x03 | 0x08 | 00000000 | *00000000* |
  50855936 = 11000010000000000000000000
  
  Little endian
  Address  0	 1	 2	 3
  Data	  00	00	08	03

  Big endian
  Address	 0   1	 2	 3
  Data	  03	08	00	00
  */
  *big = (*big >> 24) | (*big << 24) | ((*big & 0x0000FF00) << 8) | ((*big & 0x00FF0000) >> 8);
}

void save_mat(float *data, int *sizes, const char *name) {
  FILE *f = fopen(name, "wb");
  fwrite(&sizes[0], sizeof(int), 1, f);
  fwrite(&sizes[1], sizeof(int), 1, f);
  fwrite(data, sizeof(float), sizes[0] * sizes[1], f);
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

void matmul(float *A, float *B, float *C, int *sizes) {
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

void broadcast_sum(float *A, float *B, float *C, int *sizes) {
  for (int i = 0; i < sizes[0]; i++) {
    for (int j = 0; j < sizes[1]; j++) {
      C[i * sizes[1] + j] = A[i * sizes[1] + j] + B[i];
    }
  }
}
float *collapse_sum(float *A, int *sizes){
  float *out = (float *)malloc(sizeof(float) * sizes[0]);
  for (int i = 0; i < sizes[0]; i++) {
    float sum = 0.0f;
    for (int j = 0; j < sizes[1]; j++) {
      sum += A[i * sizes[1] + j];
    }
    out[i] = sum;
  }
  return out;
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

float *zeros(int size) {  
  float *A = (float *)malloc(sizeof(float) * size);
  for(int i = 0; i < size; i++){
    A[i] = 0.0f;
  }
  return A;
}

float *he_init(int ins, int outs) {
  float *weights = (float *)malloc(sizeof(float) * ins * outs);
  float scale = sqrt(12.00f / (ins + outs)); // 4/ins+outs/var_prev
  srand(time(NULL));

  for (int i = 0; i < ins * outs; i++) {
    float rand_val = ((float)rand() / RAND_MAX) * 2 - 1; // random(-1, 1) var=1/3
    weights[i] = rand_val * scale;
  }
  return weights;
}

Layer create_dense(int ins, int outs) {
  Layer layer;
  layer.type = DENSE;
  layer.ins = ins;
  layer.outs = outs;
  layer.X = NULL;
  layer.W = he_init(outs, ins);
  layer.B = zeros(outs);
  layer.H = NULL;
  layer.dW = NULL;
  layer.dB = NULL;
  layer.dH = NULL;
  return layer;
}
Layer create_relu(int size) {
  Layer layer;
  layer.type = RELU;
  layer.ins = size;
  layer.outs = size;
  layer.X = NULL;
  layer.W = NULL;
  layer.B = NULL;
  layer.H = NULL;
  layer.dW = NULL;
  layer.dB = NULL;
  layer.dH = NULL;
  return layer;
}
Layer create_softmax(int size) {
  Layer layer;
  layer.type = SOFTMAX;
  layer.ins = size;
  layer.outs = size;
  layer.X = NULL;
  layer.W = NULL;
  layer.B = NULL;
  layer.H = NULL;
  layer.dW = NULL;
  layer.dB = NULL;
  layer.dH = NULL;
  return layer;
}

float *layer_forward(Layer *layer, float *input, int input_size) {
  if (layer->H != NULL) {
    free(layer->H);
    layer->H = NULL;
  }

  switch (layer->type) {
  case DENSE: {
    if (layer->X != NULL) {
      free(layer->X);
      layer->X = NULL;
    }
      
    layer->H = (float *)malloc( sizeof(float) * layer->outs * input_size);
    layer->X = (float *)malloc(sizeof(float) * layer->ins * input_size);
    memcpy(layer->X, input, sizeof(float) * layer->ins * input_size);

    int sizes[4] = {layer->outs, layer->ins, layer->ins, input_size};
    int sizes_bias[2] = {layer->outs, input_size};
    matmul(layer->W, input, layer->H, sizes);
    broadcast_sum(layer->H, layer->B, layer->H, sizes_bias);
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
  
    for (int i = 0; i < input_size; i++) {
      // Find max for numerical stability
      float max_val = -INFINITY;
      for (int j = 0; j < layer->ins; j++) {
        if (input[j * input_size + i] > max_val) {
          max_val = input[j * input_size + i];
        }
      }

      // Sum of exp(x - max_val)
      float sum = 0.0f;
      for (int j = 0; j < layer->ins; j++) {
        float val = expf(input[j * input_size + i] - max_val);
        layer->H[j * input_size + i] = val;  // Store intermediate value
        sum += val;
      }

      // Normalize to get probabilities
      if (sum > 0) {  // Add check for zero sum
        for (int j = 0; j < layer->ins; j++) {
          layer->H[j * input_size + i] /= sum;
          // Clamp to avoid exact 0 or 1
          layer->H[j * input_size + i] = fmaxf(fminf(layer->H[j * input_size + i], 0.9999f), 0.0001f);
          }
      } else {
        // If sum is 0, output uniform distribution
        for (int j = 0; j < layer->ins; j++) {
            layer->H[j * input_size + i] = 1.0f / layer->ins;
        }
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
  if (layer->dH != NULL) {
    free(layer->dH);
    layer->dH = NULL;
  }
  if (layer->dW != NULL) {
    free(layer->dW);
    layer->dW = NULL;
  }
  if (layer->dB != NULL) {
    free(layer->dB);
    layer->dB = NULL;
  }

  switch (layer->type) {
  case DENSE: {
    layer->dH = (float *)malloc(sizeof(float) * layer->ins * size_above[1]);
    layer->dW = (float *)malloc(sizeof(float) * layer->outs * layer->ins);
    layer->dB = (float *)malloc(sizeof(float) * layer->outs);

    int X_sizes[2] = {layer->ins, size_above[1]};
    float *X_T = transpose(layer->X, X_sizes);
    int sizes[4] = {size_above[0], size_above[1], X_sizes[1], X_sizes[0]};
    matmul(dH_above, X_T, layer->dW, sizes);

    int W_sizes[2] = {layer->outs, layer->ins};
    float *W_T = transpose(layer->W, W_sizes);
    int sizes_dH[4] = {layer->ins, layer->outs, layer->outs, size_above[1]};
    matmul(W_T, dH_above, layer->dH, sizes_dH);

    layer->dB = collapse_sum(dH_above, size_above);

    free(W_T); free(X_T);
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
    float *current_grad = (float *)malloc(sizeof(float) * sizes[0] * sizes[1]);

    // Initialize gradients for cross-entropy loss with softmax
    for (int i = 0; i < sizes[1]; i++) {
        for (int j = 0; j < sizes[0]; j++) {
            // Softmax derivative with cross-entropy loss simplifies to (y_pred - y_true)
            float y_pred = previous_output[j * sizes[1] + i];
            float y_true = (j == y[i]) ? 1.0f : 0.0f;
            current_grad[j * sizes[1] + i] = (y_pred - y_true) / sizes[1];  // Divide by batch size
        }
    }

    int size_above[2] = {sizes[0], sizes[1]};
    for (int i = net->num_layers; i > 0; i--) {
        current_grad = layer_backward(&net->layers[i - 1], current_grad, size_above);
        size_above[0] = net->layers[i - 1].ins;
    }
}

int *get_prediction(float *h, int *sizes) {
  int *predictions = (int *)malloc(sizeof(int) * sizes[0] * sizes[1]);
  for (int i = 0; i < sizes[1]; i++) {
    float pred = 0;
    int max = 0;
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

void update_weights(Network *net, float lr) {
  for (int i = 0; i < net->num_layers; i++) {
    Layer layer = net->layers[i];
    if (layer.type == DENSE) {
      for (int j = 0; j < layer.outs; j++) {
        layer.B[j] -= lr * layer.dB[j];
        for (int k = 0; k < layer.ins; k++) {
          layer.W[j * layer.ins + k] -= lr * layer.dW[j * layer.ins + k];
        }
      }
    }
  }
}

int main() {
  FILE *train_images = fopen(DATA_DIR "train-images-idx3-ubyte", "rb");
  FILE *train_labels = fopen(DATA_DIR "train-labels-idx1-ubyte", "rb");
  // FILE *test_images = fopen(DATA_DIR "t10k-images-idx3-ubyte", "rb");
  // FILE *test_labels = fopen(DATA_DIR "t10k-labels-idx3-ubyte", "rb");

  uint32_t magic_number; // read the first 32 bits/4 bytes
  fread(&magic_number, sizeof(uint32_t), 1, train_images);
  uint8_t data_type = (magic_number >> 16) & 0xFF; // 3rd byte
  uint8_t dimension = (magic_number >> 24) & 0xFF; // 4th byte

  int data_size[dimension];
  uint32_t size;
  size_t total_size = 1;
  for (int i = 0; i < dimension; i++) {
    fread(&size, sizeof(uint32_t), 1, train_images);
    lilfbig(&size);
    total_size *= size;
    data_size[i] = (int)size;
    printf("dim_%d = %d\n", i, size);
  }
  int split = 60;
  total_size /= split;
  data_size[0] /= split;
  printf("data size (1/%d): %lu\n",split, total_size);

  float *train_data = (float *)malloc(sizeof(float) * total_size);
  uint8_t pixel;
  for (int i = 0; i < total_size; i++) {
    fread(&pixel, sizeof(uint8_t), 1, train_images);
    train_data[i] = (float)pixel / 255.0f; // normalize to 0-1
  }
  fclose(train_images);

  fseek(train_labels, 8, SEEK_SET);
  int *train_classes = (int *)malloc(sizeof(int) * data_size[0]);
  uint8_t label;
  for (int i = 0; i < data_size[0]; i++) {
    fread(&label, 1, 1, train_labels);
    train_classes[i] = (int)label;
  }
  fclose(train_labels);

  int train_size[2] = {data_size[0], data_size[1] * data_size[2]};
  float *input = transpose(train_data, train_size); // to column-major

  Network net = create_network();
  add_layer(&net, create_dense(train_size[1], 256));
  add_layer(&net, create_relu(256));
  add_layer(&net, create_dense(256, 128));
  add_layer(&net, create_relu(128));
  add_layer(&net, create_dense(128, 64));
  add_layer(&net, create_relu(64));
  add_layer(&net, create_dense(64, 10));
  add_layer(&net, create_softmax(10));

  int out_size[2] = {10, train_size[0]};
  int iters = 100;
  float lr = 1;
  
  for (int i = 0; i < iters; i++) {
    float *h = forward(&net, input, data_size[0]);
    float loss = calc_loss(h, train_classes, out_size);
    printf("iter=%d -- loss=%.4f\n", i, loss);
    backward(&net, loss, out_size, train_classes);    
    update_weights(&net, lr);
  }

  return 0;
}
