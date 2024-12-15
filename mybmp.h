#ifndef MYBMP_H
#define MYBMP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void draw(uint8_t width, uint8_t length, uint8_t channels, uint8_t *data);

#ifdef __cplusplus
}
#endif

#endif
