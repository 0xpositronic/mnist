/*
0-14 = BMP file header (we only fill first 3)
14-54 = 40 byte long DIB(info) header.
*/
#include "mybmp.h"
#include <stdint.h>
#include <stdio.h> // for FILE
#include <stdlib.h>
#include <string.h>  // for memcpy

void draw(uint8_t width, uint8_t length, uint8_t channels, uint8_t *data) {
  int file_size = width * length * channels + 54;
  unsigned char *bmp = (unsigned char *)malloc(file_size);

  for (int i = 0; i < file_size; i++) {
    bmp[i] = 0;
  }

  bmp[0] = 'B';
  bmp[1] = 'M';
  bmp[2] = file_size;

  bmp[10] = 54; // total header size
  bmp[14] = 40; // next 40 bytes will contain info about the image (DIB header)

  bmp[18] = width;
  bmp[22] = length;
  bmp[26] = 1;            // plane has to be 1
  bmp[28] = 8 * channels; // bits per pixel/

  memcpy(bmp + 54, data, width * length * channels);

  FILE *file = fopen("out.bmp", "wb");
  fwrite(bmp, 1, file_size, file); // 1 = size of each element in bytes
  fclose(file);
}
