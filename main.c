/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include "adxl345.h"
#include "as5600.h"

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
/* ---- settings you might change ---- */
#define USE_MCP9600      0        /* 0 = thermocouple off (sensor broken). Set to 1 when the new one is wired in. */
#define MCP9600_ADDR     (0x67 << 1)

/* ---- 800 Hz accelerometer timing ---- */
#define SAMPLE_US        1250     /* 800 Hz = one sample every 1250 microseconds */
#define POLL_US          2000     /* empty the ADXL345's FIFO buffer every 2 ms */
#define AS5600_EVERY_US  5000     /* filament wheel read 200 times per second */
#define TEMP_EVERY_US    100000   /* temperature read 10 times per second */
#define I2C_ERR_RESET    3        /* this many failed reads in a row -> reset the I2C bus */

/* ADXL345 registers used here (the rest are in adxl345.h) */
#define ADXL_REG_BW_RATE      0x2C
#define ADXL_REG_POWER_CTL    0x2D
#define ADXL_REG_DATA_FORMAT  0x31
#define ADXL_REG_FIFO_CTL     0x38
#define ADXL_REG_FIFO_STATUS  0x39

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/

/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

I2C_HandleTypeDef hi2c1;
I2C_HandleTypeDef hi2c3;

UART_HandleTypeDef huart2;

/* USER CODE BEGIN PV */
as5600_tracker_t fil = {0};   /* AS5600 running filament total */
int32_t temp_c100 = 0;        /* last thermocouple reading, in 1/100 deg C */
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_I2C1_Init(void);
static void MX_USART2_UART_Init(void);
static void MX_I2C3_Init(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
/* printf goes to USART2 = the USB serial port (used for the startup messages) */
int __io_putchar(int ch)
{
  HAL_UART_Transmit(&huart2, (uint8_t *)&ch, 1, 10);
  return ch;
}

/* Sends a whole line at once (much faster than printf one character at a time) */
static void uart_send(const char *s, int len)
{
  HAL_UART_Transmit(&huart2, (uint8_t *)s, (uint16_t)len, 50);
}

/* ---- Microsecond clock from the CPU's cycle counter ----
   Counts up in microseconds and wraps around every ~71 minutes;
   the laptop logger unwraps it. Call us_now() at least every 50 s. */
static void us_init(void)
{
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

static uint32_t us_now(void)
{
  static uint32_t last_cyc = 0, us = 0, rem = 0;
  uint32_t cyc = DWT->CYCCNT;
  rem += cyc - last_cyc;
  last_cyc = cyc;
  uint32_t per_us = SystemCoreClock / 1000000U;   /* 84 cycles per microsecond */
  us += rem / per_us;
  rem %= per_us;
  return us;
}

/* ---- ADXL345 extras for 800 Hz + FIFO ---- */
static HAL_StatusTypeDef adxl_write(uint8_t reg, uint8_t val)
{
  return HAL_I2C_Mem_Write(&hi2c1, ADXL345_ADDR, reg, I2C_MEMADD_SIZE_8BIT, &val, 1, 5);
}

/* How many samples are waiting in the ADXL345's buffer (0 to 32) */
static HAL_StatusTypeDef adxl_fifo_count(uint8_t *n)
{
  uint8_t v = 0;
  HAL_StatusTypeDef s = HAL_I2C_Mem_Read(&hi2c1, ADXL345_ADDR, ADXL_REG_FIFO_STATUS,
                                         I2C_MEMADD_SIZE_8BIT, &v, 1, 5);
  *n = v & 0x3F;
  return s;
}

/* Writes the temperature for the end of a data line: "24.50\r\n", or "nan\r\n" if not available */
static int fmt_temp(char *b, int size, int32_t v, int ok)
{
  if (!ok) return snprintf(b, size, "nan\r\n");
  const char *sign = (v < 0) ? "-" : "";
  if (v < 0) v = -v;
  return snprintf(b, size, "%s%ld.%02ld\r\n", sign, (long)(v / 100), (long)(v % 100));
}

static void delay_us(uint32_t d)
{
  uint32_t t = us_now();
  while ((uint32_t)(us_now() - t) < d) { }
}

/* Puts the ADXL345 in 800 Hz, +/-16 g, FIFO stream mode. Returns 1 if every write worked. */
static int adxl_configure(void)
{
  int ok = 1;
  ok &= (adxl_write(ADXL_REG_POWER_CTL, 0x00) == HAL_OK);    /* standby while configuring */
  ok &= (adxl_write(ADXL_REG_DATA_FORMAT, 0x0B) == HAL_OK);  /* full resolution, +/-16 g: still 256 counts per g */
  ok &= (adxl_write(ADXL_REG_BW_RATE, 0x0D) == HAL_OK);      /* 800 Hz output rate */
  ok &= (adxl_write(ADXL_REG_FIFO_CTL, 0x00) == HAL_OK);     /* bypass first = empty the buffer */
  ok &= (adxl_write(ADXL_REG_FIFO_CTL, 0x80) == HAL_OK);     /* stream mode: keeps the newest 32 samples */
  ok &= (adxl_write(ADXL_REG_POWER_CTL, 0x08) == HAL_OK);    /* start measuring */
  return ok;
}

/* ---- I2C bus recovery ----
   If a transfer gets corrupted (noise on the cable, a sensor holding the clock),
   the STM32's I2C peripheral can stay stuck "busy" and every read fails until reset.
   This resets the peripheral and clocks the bus free, without restarting the board. */
static void i2c1_recover(void)
{
  hi2c1.Instance->CR1 |= I2C_CR1_SWRST;       /* reset the I2C peripheral's internal state */
  hi2c1.Instance->CR1 &= ~I2C_CR1_SWRST;
  HAL_I2C_DeInit(&hi2c1);                     /* releases PB8 (SCL) and PB9 (SDA) */

  GPIO_InitTypeDef g = {0};
  __HAL_RCC_GPIOB_CLK_ENABLE();
  g.Pin = GPIO_PIN_8 | GPIO_PIN_9;
  g.Mode = GPIO_MODE_OUTPUT_OD;
  g.Pull = GPIO_NOPULL;
  g.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_8 | GPIO_PIN_9, GPIO_PIN_SET);
  HAL_GPIO_Init(GPIOB, &g);

  /* A sensor stuck mid-byte holds SDA low: up to 9 clock pulses let it finish and let go */
  for (int i = 0; i < 9 && HAL_GPIO_ReadPin(GPIOB, GPIO_PIN_9) == GPIO_PIN_RESET; i++) {
    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_8, GPIO_PIN_RESET); delay_us(5);
    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_8, GPIO_PIN_SET);   delay_us(5);
  }
  /* STOP condition: SDA goes low -> high while SCL is high */
  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_9, GPIO_PIN_RESET); delay_us(5);
  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_8, GPIO_PIN_SET);   delay_us(5);
  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_9, GPIO_PIN_SET);   delay_us(5);

  HAL_I2C_Init(&hi2c1);                       /* puts PB8/PB9 back in I2C mode */
}

#if USE_MCP9600
/* MCP9600: write the register number, then read 2 bytes (separate steps work
   more reliably on this chip than a combined HAL_I2C_Mem_Read) */
static HAL_StatusTypeDef mcp9600_read_reg(uint8_t reg, uint8_t b[2])
{
  HAL_StatusTypeDef s = HAL_I2C_Master_Transmit(&hi2c1, MCP9600_ADDR, &reg, 1, 50);
  if (s != HAL_OK) return s;
  return HAL_I2C_Master_Receive(&hi2c1, MCP9600_ADDR, b, 2, 50);
}

/* Thermocouple tip temperature in 1/100 deg C (2450 = 24.50 C) */
static HAL_StatusTypeDef mcp9600_read_c100(int32_t *c100)
{
  uint8_t b[2] = {0};
  HAL_StatusTypeDef s = mcp9600_read_reg(0x00, b);
  if (s == HAL_OK) {
    int16_t raw = (int16_t)((b[0] << 8) | b[1]);   /* 1 count = 1/16 C */
    *c100 = ((int32_t)raw * 100) / 16;
  }
  return s;
}
#endif
/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_I2C1_Init();
  MX_USART2_UART_Init();
  MX_I2C3_Init();
  /* USER CODE BEGIN 2 */
  setvbuf(stdout, NULL, _IONBF, 0);
  us_init();
  printf("\r\n--- 800 Hz firmware ---\r\n");

  printf("I2C1 scan:");
  for (uint8_t a = 1; a < 128; a++) {
    if (HAL_I2C_IsDeviceReady(&hi2c1, a << 1, 2, 10) == HAL_OK) {
      printf(" 0x%02X", a);
    }
  }
  printf("   (expect 0x36 0x53, plus 0x67 if the MCP9600 is plugged in)\r\n");

  /* ---- ADXL345: 800 Hz, +/-16 g, FIFO stream mode ---- */
  uint8_t id = 0;
  adxl345_read_devid(&hi2c1, &id);
  printf("ADXL345 DEVID = 0x%02X (expect 0xE5)\r\n", id);
  int adxl_ok = adxl_configure();
  printf("ADXL345 800 Hz + FIFO setup: %s\r\n", adxl_ok ? "OK" : "FAILED");

  /* ---- AS5600 ---- */
  uint8_t st = 0, agc = 0;
  uint16_t mag = 0;
  as5600_read_status(&hi2c1, &st);
  as5600_read_agc(&hi2c1, &agc);
  as5600_read_magnitude(&hi2c1, &mag);
  printf("AS5600 STATUS=0x%02X magnet=%s weak=%s strong=%s AGC=%u MAG=%u\r\n",
         st,
         (st & AS5600_STATUS_MD) ? "YES" : "no",
         (st & AS5600_STATUS_ML) ? "YES" : "no",
         (st & AS5600_STATUS_MH) ? "YES" : "no",
         agc, mag);

  /* ---- MCP9600 ---- */
  int temp_ok = 0;
#if USE_MCP9600
  uint8_t mb[2] = {0};
  mcp9600_read_reg(0x20, mb);     /* first one after the scan can fail */
  mcp9600_read_reg(0x20, mb);
  printf("MCP9600 ID = 0x%02X (expect 0x40)\r\n", mb[0]);
  uint8_t w1[2] = {0x05, 0x00};   /* thermocouple type K */
  uint8_t w2[2] = {0x06, 0x00};   /* normal running mode */
  HAL_I2C_Master_Transmit(&hi2c1, MCP9600_ADDR, w1, 2, 50);
  HAL_I2C_Master_Transmit(&hi2c1, MCP9600_ADDR, w2, 2, 50);
  HAL_Delay(300);
  temp_ok = (mcp9600_read_c100(&temp_c100) == HAL_OK);
  {
    char tb[16];
    fmt_temp(tb, sizeof tb, temp_c100, temp_ok);
    printf("MCP9600 temp = %s", tb);
  }
#else
  printf("MCP9600 off (USE_MCP9600 = 0): temp_c will be nan\r\n");
#endif

  printf("t_us,seq,x,y,z,fil_counts,temp_c\r\n");

  char line[96];
  uint32_t seq = 0;                       /* sample number, lets the logger spot lost samples */
  uint32_t t0 = us_now();
  uint32_t last_poll = t0, last_fil = t0, last_temp = t0, last_press = t0, last_batch = t0;
  uint8_t i2c_errs = 0;                   /* failed reads in a row */
  uint32_t bus_resets = 0;
  adxl_write(ADXL_REG_FIFO_CTL, 0x00);    /* start the stream with an empty buffer */
  adxl_write(ADXL_REG_FIFO_CTL, 0x80);
  (void)last_temp;
  /* USER CODE END 2 */

  /* Initialize leds */
  BSP_LED_Init(LED2);

  /* Initialize USER push-button, will be used to trigger an interrupt each time it's pressed.*/
  BSP_PB_Init(BUTTON_USER, BUTTON_MODE_EXTI);

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
    uint32_t now = us_now();

    /* Blue USER button (PC13) zeros the filament total.
       No HAL_Delay here: a pause longer than 40 ms would overflow the ADXL345 buffer. */
    if (HAL_GPIO_ReadPin(GPIOC, GPIO_PIN_13) == GPIO_PIN_RESET && (uint32_t)(now - last_press) > 300000U) {
      last_press = now;
      fil.total_counts = 0;
      uart_send("--- zeroed ---\r\n", 16);
    }

    /* Filament wheel, 200 times per second */
    if ((uint32_t)(now - last_fil) >= AS5600_EVERY_US) {
      last_fil = now;
      if (as5600_update(&hi2c1, &fil) != HAL_OK) {
        uart_send("read error as5600\r\n", 19);
        i2c_errs++;
      }
    }

#if USE_MCP9600
    /* Temperature, 10 times per second */
    if ((uint32_t)(now - last_temp) >= TEMP_EVERY_US) {
      last_temp = now;
      temp_ok = (mcp9600_read_c100(&temp_c100) == HAL_OK);
      if (!temp_ok) i2c_errs++;
    }
#endif

    /* Accelerometer: every 2 ms, send everything waiting in the FIFO (usually 1 or 2 samples) */
    if ((uint32_t)(now - last_poll) >= POLL_US) {
      last_poll = now;
      uint8_t n = 0;
      if (adxl_fifo_count(&n) != HAL_OK) {
        uart_send("read error adxl fifo\r\n", 22);
        i2c_errs++;
      } else if (n > 0) {
        uint32_t t_batch = us_now();
        if (n >= 32) {
          /* Buffer was full: some samples were lost. Skip seq ahead by about how many,
             so the logger can count them. */
          uint32_t expected = (t_batch - last_batch) / SAMPLE_US;
          if (expected > n) seq += expected - n;
          uart_send("fifo overflow\r\n", 15);
        }
        last_batch = t_batch;
        for (uint8_t i = 0; i < n; i++) {
          int16_t x, y, z;
          if (adxl345_read_xyz(&hi2c1, &x, &y, &z) != HAL_OK) {   /* each read takes the oldest sample out */
            uart_send("read error adxl\r\n", 17);
            i2c_errs++;
            break;
          }
          i2c_errs = 0;                   /* a good accelerometer read means the bus is fine */
          /* oldest sample in the batch is the earliest one */
          uint32_t t = t_batch - (uint32_t)(n - 1 - i) * SAMPLE_US;
          int len = snprintf(line, sizeof line, "%lu,%lu,%d,%d,%d,%ld,",
                             (unsigned long)t, (unsigned long)seq, x, y, z, (long)fil.total_counts);
          len += fmt_temp(line + len, (int)sizeof line - len, temp_c100, temp_ok);
          uart_send(line, len);
          seq++;
        }
      }
    }

    /* Bus stuck? Reset it and carry on instead of failing forever */
    if (i2c_errs >= I2C_ERR_RESET) {
      i2c_errs = 0;
      bus_resets++;
      i2c1_recover();
      int ok = adxl_configure();              /* also empties the FIFO */
      uint32_t t_now = us_now();
      seq += (t_now - last_batch) / SAMPLE_US; /* count the samples missed while stuck */
      last_batch = t_now;
      last_poll = t_now;
      int len = snprintf(line, sizeof line, "i2c bus reset %lu (adxl %s)\r\n",
                         (unsigned long)bus_resets, ok ? "ok" : "FAILED");
      uart_send(line, len);
    }
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE3);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 16;
  RCC_OscInitStruct.PLL.PLLN = 336;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV4;
  RCC_OscInitStruct.PLL.PLLQ = 2;
  RCC_OscInitStruct.PLL.PLLR = 2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief I2C1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_I2C1_Init(void)
{

  /* USER CODE BEGIN I2C1_Init 0 */

  /* USER CODE END I2C1_Init 0 */

  /* USER CODE BEGIN I2C1_Init 1 */

  /* USER CODE END I2C1_Init 1 */
  hi2c1.Instance = I2C1;
  hi2c1.Init.ClockSpeed = 400000;
  hi2c1.Init.DutyCycle = I2C_DUTYCYCLE_2;
  hi2c1.Init.OwnAddress1 = 0;
  hi2c1.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
  hi2c1.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
  hi2c1.Init.OwnAddress2 = 0;
  hi2c1.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
  hi2c1.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
  if (HAL_I2C_Init(&hi2c1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN I2C1_Init 2 */

  /* USER CODE END I2C1_Init 2 */

}

/**
  * @brief I2C3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_I2C3_Init(void)
{

  /* USER CODE BEGIN I2C3_Init 0 */

  /* USER CODE END I2C3_Init 0 */

  /* USER CODE BEGIN I2C3_Init 1 */

  /* USER CODE END I2C3_Init 1 */
  hi2c3.Instance = I2C3;
  hi2c3.Init.ClockSpeed = 100000;
  hi2c3.Init.DutyCycle = I2C_DUTYCYCLE_2;
  hi2c3.Init.OwnAddress1 = 0;
  hi2c3.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
  hi2c3.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
  hi2c3.Init.OwnAddress2 = 0;
  hi2c3.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
  hi2c3.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
  if (HAL_I2C_Init(&hi2c3) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN I2C3_Init 2 */

  /* USER CODE END I2C3_Init 2 */

}

/**
  * @brief USART2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART2_UART_Init(void)
{

  /* USER CODE BEGIN USART2_Init 0 */

  /* USER CODE END USART2_Init 0 */

  /* USER CODE BEGIN USART2_Init 1 */

  /* USER CODE END USART2_Init 1 */
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 921600;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART2_Init 2 */

  /* USER CODE END USART2_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
