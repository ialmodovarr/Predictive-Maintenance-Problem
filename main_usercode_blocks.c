/* =====================================================================
   ADXL345 streaming: the four blocks to paste into a NEW CubeMX project's
   Core/Src/main.c. Each block goes BETWEEN the matching pair of comments
   that CubeMX already wrote, for example between
       / * USER CODE BEGIN Includes * /   and   / * USER CODE END Includes * /
   Do NOT copy the BEGIN/END lines themselves and do NOT rename them:
   CubeMX only keeps code that sits between its exact markers.
   (Spaces added inside the comment markers above so this note stays a comment.)
   ===================================================================== */


/* ---- 1. between USER CODE BEGIN Includes / END Includes ---- */
#include <stdio.h>
#include "adxl345.h"


/* ---- 2. between USER CODE BEGIN 0 / END 0 ---- */
/* Sends printf output to USART2 = the USB serial port on the Nucleo. */
int __io_putchar(int ch)
{
  HAL_UART_Transmit(&huart2, (uint8_t *)&ch, 1, 10);
  return ch;
}


/* ---- 3. between USER CODE BEGIN 2 / END 2 (runs once at startup) ---- */
  printf("\r\nI2C scan:");
  for (uint8_t a = 1; a < 128; a++) {
    if (HAL_I2C_IsDeviceReady(&hi2c1, a << 1, 2, 10) == HAL_OK) {
      printf(" 0x%02X", a);
    }
  }
  printf("\r\n");

  uint8_t id = 0;
  adxl345_read_devid(&hi2c1, &id);
  printf("ADXL345 DEVID = 0x%02X (expect 0xE5)\r\n", id);

  if (adxl345_init(&hi2c1) != HAL_OK) {
    printf("adxl345_init FAILED\r\n");
  }
  printf("t_ms,x,y,z\r\n");
  uint32_t next_sample = HAL_GetTick();


/* ---- 4. between USER CODE BEGIN 3 / END 3 (inside while (1)) ---- */
    /* One sample every 5 ms = 200 samples per second. */
    if ((int32_t)(HAL_GetTick() - next_sample) >= 0) {
      next_sample += 5;
      int16_t x, y, z;
      if (adxl345_read_xyz(&hi2c1, &x, &y, &z) == HAL_OK) {
        printf("%lu,%d,%d,%d\r\n", (unsigned long)HAL_GetTick(), x, y, z);
      } else {
        printf("read error\r\n");
      }
    }
