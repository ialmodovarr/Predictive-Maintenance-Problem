#include "adxl345.h"

static HAL_StatusTypeDef write_reg(I2C_HandleTypeDef *hi2c, uint8_t reg, uint8_t val)
{
    return HAL_I2C_Mem_Write(hi2c, ADXL345_ADDR, reg, I2C_MEMADD_SIZE_8BIT, &val, 1, ADXL345_TIMEOUT_MS);
}

HAL_StatusTypeDef adxl345_read_devid(I2C_HandleTypeDef *hi2c, uint8_t *id)
{
    return HAL_I2C_Mem_Read(hi2c, ADXL345_ADDR, ADXL345_REG_DEVID, I2C_MEMADD_SIZE_8BIT, id, 1, ADXL345_TIMEOUT_MS);
}

HAL_StatusTypeDef adxl345_init(I2C_HandleTypeDef *hi2c)
{
    HAL_StatusTypeDef s;

    /* DATA_FORMAT: FULL_RES = 1, range +/-4 g -> about 256 counts per g */
    s = write_reg(hi2c, ADXL345_REG_DATA_FORMAT, 0x09);
    if (s != HAL_OK) return s;

    /* BW_RATE: 200 Hz output rate (0x0B), matches the 200 Hz stream in main.c */
    s = write_reg(hi2c, ADXL345_REG_BW_RATE, 0x0B);
    if (s != HAL_OK) return s;

    /* POWER_CTL: Measure bit on. Without this every reading is 0. */
    return write_reg(hi2c, ADXL345_REG_POWER_CTL, 0x08);
}

HAL_StatusTypeDef adxl345_read_xyz(I2C_HandleTypeDef *hi2c, int16_t *x, int16_t *y, int16_t *z)
{
    uint8_t b[6];
    HAL_StatusTypeDef s = HAL_I2C_Mem_Read(hi2c, ADXL345_ADDR, ADXL345_REG_DATAX0,
                                           I2C_MEMADD_SIZE_8BIT, b, 6, ADXL345_TIMEOUT_MS);
    if (s != HAL_OK) return s;

    /* Little-endian: low byte first */
    *x = (int16_t)((b[1] << 8) | b[0]);
    *y = (int16_t)((b[3] << 8) | b[2]);
    *z = (int16_t)((b[5] << 8) | b[4]);
    return HAL_OK;
}
