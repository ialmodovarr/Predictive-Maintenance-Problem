#ifndef AS5600_H
#define AS5600_H

#include "stm32f4xx_hal.h"   /* Nucleo-F446RE */

/* Fixed address, cannot be changed on the AS5600 */
#define AS5600_ADDR            (0x36 << 1)

/* I2C timeout in ms. Short, so a failed read can't stall the 800 Hz accelerometer stream */
#define AS5600_TIMEOUT_MS      5

#define AS5600_REG_STATUS      0x0B   /* bit5 MD = magnet detected, bit4 ML = too weak, bit3 MH = too strong */
#define AS5600_REG_RAW_ANGLE   0x0C   /* 0x0C high byte (bits 11:8), 0x0D low byte */
#define AS5600_REG_AGC         0x1A   /* gain: at 3.3 V range is 0-128, ~64 = gap about right, 128 = maxed (field weak) */
#define AS5600_REG_MAGNITUDE   0x1B   /* 0x1B high, 0x1C low */

#define AS5600_STATUS_MH       (1 << 3)
#define AS5600_STATUS_ML       (1 << 4)
#define AS5600_STATUS_MD       (1 << 5)

#define AS5600_COUNTS_PER_TURN 4096

/* Wheel circumference in micrometers: pi * 16 mm = 50.27 mm.
   Tune this after the 100 mm hand-pull test:
   new value = old value * (100 / mm the sensor reported). */
#define AS5600_UM_PER_TURN     50270

/* Tracks total wheel rotation across the 4095 -> 0 wraparound */
typedef struct {
    uint16_t last_raw;
    int32_t  total_counts;   /* signed: goes negative if filament moves backward (retraction) */
    uint8_t  started;
} as5600_tracker_t;

HAL_StatusTypeDef as5600_read_status(I2C_HandleTypeDef *hi2c, uint8_t *status);
HAL_StatusTypeDef as5600_read_raw(I2C_HandleTypeDef *hi2c, uint16_t *raw);
HAL_StatusTypeDef as5600_read_agc(I2C_HandleTypeDef *hi2c, uint8_t *agc);
HAL_StatusTypeDef as5600_read_magnitude(I2C_HandleTypeDef *hi2c, uint16_t *mag);

/* Reads the angle and adds the change since last call to t->total_counts */
HAL_StatusTypeDef as5600_update(I2C_HandleTypeDef *hi2c, as5600_tracker_t *t);

/* Filament distance in micrometers from a count total */
int32_t as5600_counts_to_um(int32_t counts);

#endif
