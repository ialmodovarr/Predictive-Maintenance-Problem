#ifndef ADXL345_H
#define ADXL345_H

#include "stm32f4xx_hal.h"   /* Nucleo-F446RE. Change if you use a different STM32 family. */

/* SDO/ALT pin low or unconnected -> 0x53. If the I2C scan shows 0x1D, use (0x1D << 1). */
#define ADXL345_ADDR            (0x53 << 1)

/* I2C timeout in ms. A good read takes ~0.25 ms at 400 kHz. Keep this short: while a failed
   read waits, the 800 Hz FIFO (32 samples = ~40 ms) can fill up and samples get lost. */
#define ADXL345_TIMEOUT_MS      5

#define ADXL345_REG_DEVID       0x00
#define ADXL345_REG_BW_RATE     0x2C
#define ADXL345_REG_POWER_CTL   0x2D
#define ADXL345_REG_DATA_FORMAT 0x31
#define ADXL345_REG_DATAX0      0x32

HAL_StatusTypeDef adxl345_read_devid(I2C_HandleTypeDef *hi2c, uint8_t *id);
HAL_StatusTypeDef adxl345_init(I2C_HandleTypeDef *hi2c);
HAL_StatusTypeDef adxl345_read_xyz(I2C_HandleTypeDef *hi2c, int16_t *x, int16_t *y, int16_t *z);

#endif
