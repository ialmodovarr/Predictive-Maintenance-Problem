#include "as5600.h"

static HAL_StatusTypeDef read_regs(I2C_HandleTypeDef *hi2c, uint8_t reg, uint8_t *buf, uint16_t len)
{
    return HAL_I2C_Mem_Read(hi2c, AS5600_ADDR, reg, I2C_MEMADD_SIZE_8BIT, buf, len, AS5600_TIMEOUT_MS);
}

HAL_StatusTypeDef as5600_read_status(I2C_HandleTypeDef *hi2c, uint8_t *status)
{
    return read_regs(hi2c, AS5600_REG_STATUS, status, 1);
}

HAL_StatusTypeDef as5600_read_raw(I2C_HandleTypeDef *hi2c, uint16_t *raw)
{
    uint8_t b[2];
    HAL_StatusTypeDef s = read_regs(hi2c, AS5600_REG_RAW_ANGLE, b, 2);
    if (s != HAL_OK) return s;
    *raw = (uint16_t)(((b[0] & 0x0F) << 8) | b[1]);   /* 12-bit, high byte first */
    return HAL_OK;
}

HAL_StatusTypeDef as5600_read_agc(I2C_HandleTypeDef *hi2c, uint8_t *agc)
{
    return read_regs(hi2c, AS5600_REG_AGC, agc, 1);
}

HAL_StatusTypeDef as5600_read_magnitude(I2C_HandleTypeDef *hi2c, uint16_t *mag)
{
    uint8_t b[2];
    HAL_StatusTypeDef s = read_regs(hi2c, AS5600_REG_MAGNITUDE, b, 2);
    if (s != HAL_OK) return s;
    *mag = (uint16_t)(((b[0] & 0x0F) << 8) | b[1]);
    return HAL_OK;
}

HAL_StatusTypeDef as5600_update(I2C_HandleTypeDef *hi2c, as5600_tracker_t *t)
{
    uint16_t raw;
    HAL_StatusTypeDef s = as5600_read_raw(hi2c, &raw);
    if (s != HAL_OK) return s;

    if (!t->started) {
        t->last_raw = raw;
        t->total_counts = 0;
        t->started = 1;
        return HAL_OK;
    }

    /* Shortest way around the circle. Works as long as the wheel turns
       less than half a turn (~25 mm of filament) between reads. At 10 Hz
       that allows up to 250 mm/s, far faster than any extrusion or retract. */
    int32_t delta = (int32_t)raw - (int32_t)t->last_raw;
    if (delta >  AS5600_COUNTS_PER_TURN / 2) delta -= AS5600_COUNTS_PER_TURN;
    if (delta < -AS5600_COUNTS_PER_TURN / 2) delta += AS5600_COUNTS_PER_TURN;

    t->total_counts += delta;
    t->last_raw = raw;
    return HAL_OK;
}

int32_t as5600_counts_to_um(int32_t counts)
{
    /* 64-bit math so long prints don't overflow */
    return (int32_t)(((int64_t)counts * AS5600_UM_PER_TURN) / AS5600_COUNTS_PER_TURN);
}
