#include "config.h"
#include <stdint.h>
#include <stdio.h>

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

void save_mat(float *data, char *name, int rows, int cols) {
  FILE *f = fopen(name, "wb");
  fwrite(&rows, sizeof(int), 1, f);
  fwrite(&cols, sizeof(int), 1, f);
  fwrite(data, sizeof(float), rows * cols, f);
  fclose(f);
  printf("saved %s with sizes(%d, %d)\n", name, rows, cols);
}

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

int main() {
  size_t e;
  FILE *train_data = fopen(DATA_DIR "train-images-idx3-ubyte", "rb");
  FILE *train_labels = fopen(DATA_DIR "train-labels-idx1-ubyte", "rb");

  uint32_t magic_number; // read the first 32 bits/4 bytes
  e = fread(&magic_number, sizeof(uint32_t), 1, train_data);
  //uint8_t data_type = (magic_number >> 16) & 0xFF; // 3rd byte
  uint8_t n_dimensions = (magic_number >> 24) & 0xFF; // 4th byte
  
  int data_size[n_dimensions];
  uint32_t size;
  size_t total_size = 1;
  for (int i = 0; i < n_dimensions; i++) {
    e = fread(&size, sizeof(uint32_t), 1, train_data);
    lilfbig(&size);
    total_size *= size;
    data_size[i] = (int)size;
    printf("dim_%d = %d\n", i, size);
  }

  int scale = 60;
  total_size /= scale;
  data_size[0] /= scale;
  printf("data size (1/%d): %lu\n",scale, total_size);

  float *batch_data = (float *)malloc(sizeof(float) * total_size);
  uint8_t pixel;
  for (int i = 0; i < total_size; i++) {
    e = fread(&pixel, sizeof(uint8_t), 1, train_data);
    batch_data[i] = (float)pixel / 255.0f; // to 0-1
  }
  fclose(train_data);
  int image_size = data_size[1] * data_size[2];
  
  fseek(train_labels, 8, SEEK_SET);
  float *batch_labels = (float *)malloc(sizeof(float) * data_size[0]);
  uint8_t label;
  for (int i = 0; i < data_size[0]; i++) {
    e = fread(&label, 1, 1, train_labels);
    batch_labels[i] = (float)label;
  }
  fclose(train_labels);
  
  if(e!=1) printf("file error\n");
  
  save_mat(batch_data, "c_batch_data.bin", data_size[0], image_size);
  save_mat(batch_labels, "c_batch_labels.bin", data_size[0], 1);
  
  Network net = create_network();
  add_layer(&net, create_dense(image_size, 128));
  add_layer(&net, create_relu(256));
  add_layer(&net, create_dense(128, 10));
  add_layer(&net, create_softmax(10));

  int d = 0;
  for (int i = 0; i < net.n_layers; i ++) {
    Layer current = net.layers[i];
    if(current.type == DENSE) {
      char w_name[100];
      char b_name[100];
      sprintf(w_name, "c_weights_%d.bin", d);
      sprintf(b_name, "c_biases_%d.bin", d);
      save_mat(current.W, w_name, current.in_size, current.out_size);
      save_mat(current.B, b_name, 1, current.out_size);
      d++;
    }
  }
  return 0;
}
