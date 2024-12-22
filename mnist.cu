#include "config.h"
#include <stdint.h>
#include <stdio.h>
#include <time.h>

typedef enum { DENSE, RELU, SOFTMAX } LayerType;
typedef struct {
  LayerType type;
  int in_size;
  int out_size;
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
  int n_layers;
} Network;

void save_mat(float *data, const char *name, int rows, int cols) {
  FILE *f = fopen(name, "wb");
  fwrite(&rows, sizeof(int), 1, f);
  fwrite(&cols, sizeof(int), 1, f);
  fwrite(data, sizeof(float), rows * cols, f);
  fclose(f);
  printf("saved %s with sizes(%d, %d)\n", name, rows, cols);
}

void lilfbig(uint32_t *big) { // big-endian to little-endian
  /*
  big endian: first byte of the 4 byte number is on the right
  | 0x03 | 0x08 | 00000000 | 00000000 |
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

float *zeros(int size) {  
  float *A = (float *)malloc(sizeof(float) * size);
  for(int i = 0; i < size; i++){
    A[i] = 0.0f;
  }
  return A;
}

float *he_init(int rows, int cols) {
  float *weights = (float *)malloc(sizeof(float) * rows * cols);
  float scale = sqrt(12.00f / (rows + cols)); // 4/rows+cols/var_prev
  //srand(time(NULL));
  srand(42);
  for (int i = 0; i < rows * cols; i++) {
    float rand_val = ((float)rand() / RAND_MAX) * 2 - 1; // random(-1, 1) var=1/3
    weights[i] = rand_val * scale;
  }
  return weights;
}

void matmul(float *A, float *B, float *C, int in_rows, int in_cols, int out_cols) {
  for (int i = 0; i < out_cols; i++) {
    for (int j = 0; j < in_rows; j++) {
      float sum = 0;
      for (int k = 0; k < in_cols; k++) {
        sum += A[j * in_cols + k] * B[k * out_cols + i];
      }
      C[j * out_cols + i] = sum;
    }
  }
}
float *transpose(float *A, int rows, int cols) {
  float *At = (float * )malloc(sizeof(float) * rows * cols);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      At[j * rows + i] = A[i * cols + j];
    }
  }
  return At;
}
void broadcast_sum(float *A, float *B, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      A[i * cols + j] = A[i * cols + j] + B[j];
    }
  }
}
float *collapse_sum(float *A, int rows, int cols) {
  float *summed = (float *)malloc(sizeof(float) * cols);
  for (int j = 0; j < cols; j++) {
    float sum = 0.0f;
    for (int i = 0; i < rows; i++) {
      sum += A[i * cols + j];
    }
    summed[j] = sum;
  }
  return summed;
}

Network create_network() {
  Network net;
  net.layers = NULL;
  net.n_layers = 0;
  return net;
}
void add_layer(Network *net, Layer layer) {
  net->n_layers++;
  net->layers = (Layer *)realloc(net->layers, sizeof(Layer) * net->n_layers);
  net->layers[net->n_layers - 1] = layer;
}

Layer create_dense(int in_size, int out_size) {
  Layer layer;
  layer.type = DENSE;
  layer.in_size = in_size;
  layer.out_size = out_size;
  layer.X = NULL;
  layer.W = he_init(in_size, out_size);
  layer.B = zeros(out_size);
  layer.H = NULL;
  layer.dW = NULL;
  layer.dB = NULL;
  layer.dH = NULL;
  return layer;
}
Layer create_relu(int size) {
  Layer layer;
  layer.type = RELU;
  layer.in_size = size;
  layer.out_size = size;
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
  layer.in_size = size;
  layer.out_size = size;
  layer.X = NULL;
  layer.W = NULL;
  layer.B = NULL;
  layer.H = NULL;
  layer.dW = NULL;
  layer.dB = NULL;
  layer.dH = NULL;
  return layer;
}

float *layer_forward(Layer *layer, float *input, int batch_size) {
  switch (layer->type) {
  case DENSE: {    
    layer->H = (float *)malloc( sizeof(float) * layer->out_size * batch_size);
    layer->X = (float *)malloc(sizeof(float) * layer->in_size * batch_size);
    memcpy(layer->X, input, sizeof(float) * layer->in_size * batch_size);

    matmul(input, layer->W, layer->H, batch_size, layer->in_size, layer->out_size);
    broadcast_sum(layer->H, layer->B, batch_size, layer->out_size);
    break;
  }
  case RELU: {
    layer->H = (float *)malloc(sizeof(float) * batch_size * layer->out_size);
    for (int i = 0; i < batch_size * layer->in_size; i++) {
      layer->H[i] = input[i] > 0 ? input[i] : 0;
    }
    break;
  }
  case SOFTMAX: {
    layer->H = (float *)malloc(sizeof(float) * batch_size * layer->out_size);
  
    for (int i = 0; i < batch_size; i++) {
      float max_val = -INFINITY;
      for (int j = 0; j < layer->out_size; j++) { // find max
        if (input[i * layer->out_size + j] > max_val) {
          max_val = input[i * layer->out_size + j];
        }
      }
      float sum = 0.0f;
      for (int j = 0; j < layer->out_size; j++) { // sum exp(x-max)
        // doesn't change the outcome exp(a-b) = exp(a)/exp(b)
        float val = expf(input[i * layer->out_size + j] - max_val);
        layer->H[i * layer->out_size + j] = val;  // Store intermediate value
        sum += val;
      }
      for (int j = 0; j < layer->out_size; j++) {
        layer->H[i * layer->out_size + j] /= sum;
      }
    }
    break;
  }
  }
  return layer->H;
}

float *layer_backward(Layer *layer, float *dH_above, int batch_size) {
  switch (layer->type) {
  case DENSE: {
    layer->dH = (float *)malloc(sizeof(float) * layer->in_size * batch_size);
    layer->dW = (float *)malloc(sizeof(float) * layer->out_size * layer->in_size);
    layer->dB = (float *)malloc(sizeof(float) * layer->out_size);

    matmul(dH_above, transpose(layer->W, layer->in_size, layer->out_size), layer->dH, batch_size, layer->out_size, layer->in_size);    
    matmul(transpose(layer->X, batch_size, layer->in_size), dH_above, layer->dW, layer->in_size, batch_size, layer->out_size);
    layer->dB = collapse_sum(dH_above, batch_size, layer->out_size);
    break;
  }
  case RELU: {
    layer->dH = (float *)malloc(sizeof(float) * batch_size * layer->out_size);
    for (int i = 0; i < batch_size * layer->out_size; i++) {
        layer->dH[i] = layer->H[i] > 0 ? dH_above[i] : 0;
    }
    break;
  }
  case SOFTMAX: {
    break; // this is never the case
  }
  }
  return layer->dH;
}

float *forward(Network *net, float *input, int batch_size, int features, int iter = 0) {
  float *h = (float * )malloc(batch_size * features * sizeof(float));
  memcpy(h, input, batch_size * features * sizeof(float));
  for (int i = 0; i < net->n_layers; i++) {
    h = layer_forward(&net->layers[i], h, batch_size);
  }
  return h;
}

void backward(Network *net, float loss, float *y, int batch_size, int iter = 0) {
    int n_classes = net->layers[net->n_layers-1].out_size;
    float *previous_output = net->layers[net->n_layers - 1].H; // softmax output
    float *current_grad = (float *)malloc(sizeof(float) * batch_size * n_classes);
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < n_classes; j++) {
            float y_pred = previous_output[i * n_classes + j];
            float y_true = (j == y[i]) ? 1.0f : 0.0f;
            current_grad[i * n_classes + j] = (y_pred - y_true) / batch_size;
        }
    }
    for (int i = net->n_layers-1; i > 0; i--) {
        current_grad = layer_backward(&net->layers[i - 1], current_grad, batch_size); // skips softmax
    }
}

float calc_loss(float *h, float *y, int batch_size, int classes) {
  float loss = 0.0f;
  const float epsilon = 1e-7f;
  for (int i = 0; i < batch_size; i++) {
    float true_class_prob = h[i * classes + (int)y[i]];
    true_class_prob = fmaxf(true_class_prob, epsilon);
    loss -= logf(true_class_prob);
  }
  loss /= batch_size;
  return loss;
}

void update_parameters(Network *net, float lr) {
  for (int l = 0; l < net->n_layers; l++) {
    Layer layer = net->layers[l];
    if (layer.type == DENSE) {
      for (int i = 0; i < layer.out_size; i++) {
        layer.B[i] -= lr * layer.dB[i];
        for (int j = 0; j < layer.in_size; j++) {
          layer.W[j * layer.out_size + i] -= lr * layer.dW[j * layer.out_size + i];
        }
      }
    }
  }
}

float *get_prediction(float *h, int batch_size, int n_classes) {
  float *predictions = (float *)malloc(sizeof(float) * batch_size * n_classes);
  for (int i = 0; i < batch_size; i++) {
    float pred = 0;
    int max = 0;
    for (int j = 0; j < n_classes; j++) {
      if (h[i * n_classes + j] > pred) {
        pred = h[i * n_classes + j];
        max = j;
      }
    }
    predictions[i] = max;
  }
  return predictions;
}


float* read_mnist_images(const char* filename, int* n_images, int* image_size, int scale) {
  FILE *data = fopen(filename, "rb");
  if (!data) {
    printf("Error opening file: %s\n", filename);
    return NULL;
  }

  uint32_t magic_number;
  size_t e = fread(&magic_number, sizeof(uint32_t), 1, data);
  uint8_t n_dimensions = (magic_number >> 24) & 0xFF;
  
  int data_size[n_dimensions];
  uint32_t size;
  size_t total_size = 1;
  for (int i = 0; i < n_dimensions; i++) {
    e = fread(&size, sizeof(uint32_t), 1, data);
    lilfbig(&size);
    total_size *= size;
    data_size[i] = (int)size;
  }

  if (scale > 1) {
    total_size /= scale;
    data_size[0] /= scale;
  }

  float *batch_data = (float *)malloc(sizeof(float) * total_size);
  uint8_t pixel;
  for (int i = 0; i < total_size; i++) {
    e = fread(&pixel, sizeof(uint8_t), 1, data);
    batch_data[i] = (float)pixel / 255.0f;
  }
  fclose(data);

  *n_images = data_size[0];
  *image_size = data_size[1] * data_size[2];
  
  if(e != 1) printf("file read error\n");
  return batch_data;
}

float* read_mnist_labels(const char* filename, int n_images) {
  FILE *labels = fopen(filename, "rb");
  if (!labels) {
    printf("Error opening file: %s\n", filename);
    return NULL;
  }

  fseek(labels, 8, SEEK_SET);  // Skip header
  float *batch_labels = (float *)malloc(sizeof(float) * n_images);
  uint8_t label;
  for (int i = 0; i < n_images; i++) {
    size_t e = fread(&label, 1, 1, labels);
    batch_labels[i] = (float)label;
    if(e != 1) printf("label read error\n");
  }
  fclose(labels);
  return batch_labels;
}

int main() {
  int n_images, image_size;
  int scale = 6;

  float *batch_data = read_mnist_images(DATA_DIR "train-images-idx3-ubyte", &n_images, &image_size, scale);
  float *batch_labels = read_mnist_labels(DATA_DIR "train-labels-idx1-ubyte", n_images);
  
  printf("Training data size (1/%d): %d\n", scale, n_images);
  int n_classes = 10;
  
  Network net = create_network();
  add_layer(&net, create_dense(image_size, 128));
  add_layer(&net, create_relu(128));
  add_layer(&net, create_dense(128, 10));
  add_layer(&net, create_softmax(10));

  float lr = 0.1;
  int iters = 50;
  float *loss_hist = (float *)malloc(sizeof(float) * iters);
  for (int i = 0; i < iters; i++) {
    float *h = forward(&net, batch_data, n_images, image_size, i);
    float loss = calc_loss(h, batch_labels, n_images, n_classes);
    backward(&net, loss, batch_labels, n_images, i);
    update_parameters(&net, lr);

    printf("iter=%d -- loss=%.6f\n", i, loss);
    loss_hist[i] = loss;
  }
  save_mat(loss_hist, "loss.bin", iters, 1);

  
  int test_n_images, test_image_size;
  float *test_data = read_mnist_images(DATA_DIR "t10k-images-idx3-ubyte", &test_n_images, &test_image_size, 1);  // No scaling for test
  float *test_labels = read_mnist_labels(DATA_DIR "t10k-labels-idx1-ubyte", test_n_images);
  
  float *h = forward(&net, test_data, test_n_images, image_size);
  float *preds = get_prediction(h, test_n_images, n_classes);
  
  int error = 0;
  for (int i = 0; i < test_n_images; i++) {
    if (preds[i] != test_labels[i]) {
      error++;
    }
  }
  printf("Test accuracy: %.4f\n", 1.0f - (error/(float)test_n_images));

  free(batch_data);
  free(batch_labels);
  free(test_data);
  free(test_labels);
  free(preds);
  free(loss_hist);  
  return 0;
}
