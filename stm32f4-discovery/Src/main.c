/**
  ******************************************************************************
  * @file    UART_TwoBoards_ComIT/main.c
  * @brief   UART IT with partial TX/RX support, LCD heartbeat display
  ******************************************************************************
  */

#include "main.h"
#include <stdlib.h>

#define HEART_SIZE 64
#define HEART_X 200
#define HEART_Y 160
#define HEART_COLOR LCD_COLOR_RED
#define LCD_T_UPDATE 120
#define BAUD_RATE 115200
#define RXBUFFERSIZE 200

UART_HandleTypeDef UartHandle;

/* Flags */
__IO ITStatus UartTxReady = SET;
__IO ITStatus UartRxReady = SET;

/* TX buffer management */
uint8_t *tx_buffer_ptr;
uint16_t tx_buffer_size;
uint16_t tx_buffer_index;

/* RX buffer management */
uint8_t *rx_buffer_ptr;
uint16_t rx_buffer_size;
uint16_t rx_buffer_index;

/* Buffers */
uint8_t aTxBuffer[] = " ****UART_TwoBoards_ComIT**** ";
uint8_t aRxBuffer[RXBUFFERSIZE];

/* Function prototypes */
static void init_uart(void);
static void init_lcd(void);
static int uart_send(uint8_t* buffer, uint16_t size);
static int uart_rcv(uint8_t* buffer, uint16_t size);
static void lcd_animation(int index);
static void draw_heart(int cx, int cy, int size, float scale);
static void clear_heart_area(int cx, int cy, int size);
static void SystemClock_Config(void);
static void Error_Handler(void);

int main(void)
{
    HAL_Init();
    BSP_LED_Init(LED3);
    BSP_LED_Init(LED4);
    SystemClock_Config();

    init_uart();
    init_lcd();

    int animation_index = 0;
    uint8_t buffer[4];
    char bpm[4];
    int bpm_index = 0;
    uint32_t tickstart = HAL_GetTick();

    while(1) {
        uint32_t elapsed = HAL_GetTick() - tickstart;
        if(elapsed >= LCD_T_UPDATE) {
            tickstart = HAL_GetTick();
            lcd_animation(animation_index++);
        }

        if (uart_rcv(buffer, 4)) {  // ricezione parziale gestita
            for(int i = 0; i < 4; i++) {
                if (buffer[i] == '\n') {
                    bpm[bpm_index] = '\0';
                    bpm_index = 0;
                    BSP_LCD_DisplayStringAt(10, HEART_Y, (uint8_t*)bpm, LEFT_MODE);
                    BSP_LED_Toggle(LED3);
                } else if (buffer[i] >= '0' && buffer[i] <= '9') {
                    if (bpm_index < 3) bpm[bpm_index++] = buffer[i];
                }
            }
        }
    }
}

/* UART Initialization */
static void init_uart() {
    UartHandle.Instance          = USARTx;
    UartHandle.Init.BaudRate     = BAUD_RATE;
    UartHandle.Init.WordLength   = UART_WORDLENGTH_8B;
    UartHandle.Init.StopBits     = UART_STOPBITS_1;
    UartHandle.Init.Parity       = UART_PARITY_NONE;
    UartHandle.Init.HwFlowCtl    = UART_HWCONTROL_NONE;
    UartHandle.Init.Mode         = UART_MODE_TX_RX;
    UartHandle.Init.OverSampling = UART_OVERSAMPLING_16;

    if(HAL_UART_Init(&UartHandle) != HAL_OK) {
        Error_Handler();
    }

    char msg[] = "Hello\r\n";
    uart_send((uint8_t*)msg, sizeof(msg)-1);
}

/* LCD Initialization */
static void init_lcd() {
    BSP_LCD_Init();
    BSP_LCD_LayerDefaultInit(1, LCD_FRAME_BUFFER);
    BSP_LCD_SelectLayer(1);
    BSP_LCD_SetBackColor(LCD_COLOR_WHITE);
    BSP_LCD_Clear(LCD_COLOR_WHITE);
    BSP_LCD_SetTextColor(LCD_COLOR_DARKBLUE);
    BSP_LCD_SetFont(&Font16);
    BSP_LCD_DisplayStringAt(0, 10, (uint8_t*)"Current heartbeat", CENTER_MODE);
    BSP_LCD_SetFont(&LCD_DEFAULT_FONT);
    BSP_LCD_DisplayStringAt(0, HEART_Y, (uint8_t*)"bpm", CENTER_MODE);
}

/* UART send with partial handling */
static int uart_send(uint8_t* buffer, uint16_t size) {
    if (UartTxReady != SET) return 0;  // trasmissione in corso

    tx_buffer_ptr = buffer;
    tx_buffer_size = size;
    tx_buffer_index = 0;
    UartTxReady = RESET;

    if(HAL_UART_Transmit_IT(&UartHandle, &tx_buffer_ptr[tx_buffer_index], 1) != HAL_OK)
        Error_Handler();

    tx_buffer_index++;
    return 1;
}

/* UART receive with partial handling */
static int uart_rcv(uint8_t* buffer, uint16_t size) {
    if (UartRxReady != SET) return 0;  // ricezione in corso

    rx_buffer_ptr = buffer;
    rx_buffer_size = size;
    rx_buffer_index = 0;
    UartRxReady = RESET;

    if(HAL_UART_Receive_IT(&UartHandle, &rx_buffer_ptr[rx_buffer_index], 1) != HAL_OK)
        Error_Handler();

    rx_buffer_index++;
    return 1;
}

/* LCD Animation */
static void lcd_animation(int index) {
    float frames[4] = {0.50f, 0.70f, 0.80f, 0.70f};
    index = index % 4;

    if(index == 0 || index == 3) clear_heart_area(HEART_X, HEART_Y, HEART_SIZE);
    draw_heart(HEART_X, HEART_Y, HEART_SIZE, frames[index]);
    HAL_Delay(120);
}

/* Heart drawing */
static void draw_heart(int cx, int cy, int size, float scale) {
    BSP_LCD_SetTextColor(HEART_COLOR);
    for(int y=-size/2;y<size/2;y++){
        for(int x=-size/2;x<size/2;x++){
            float xf = x / ((size/2.0f) * scale);
            float yf = y / ((size/2.0f) * scale);
            float a = xf*xf + yf*yf - 1;
            if ((a*a*a - xf*xf*yf*yf*yf) <= 0) {
                BSP_LCD_FillRect(cx+x, cy-y, 1, 1);
            }
        }
    }
}

static void clear_heart_area(int cx, int cy, int size) {
    int half = size/2;
    BSP_LCD_SetTextColor(LCD_COLOR_WHITE);
    BSP_LCD_FillRect(cx-half, cy-half, size, size);
}

/* UART TX callback */
void HAL_UART_TxCpltCallback(UART_HandleTypeDef *UartHandle) {
    if(tx_buffer_index < tx_buffer_size) {
        if(HAL_UART_Transmit_IT(UartHandle, &tx_buffer_ptr[tx_buffer_index], 1) != HAL_OK)
            Error_Handler();
        tx_buffer_index++;
    } else {
        UartTxReady = SET;
    }
}

/* UART RX callback */
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *UartHandle) {
    if(rx_buffer_index < rx_buffer_size) {
        if(HAL_UART_Receive_IT(UartHandle, &rx_buffer_ptr[rx_buffer_index], 1) != HAL_OK)
            Error_Handler();
        rx_buffer_index++;
    } else {
        UartRxReady = SET;
    }
}

void HAL_UART_ErrorCallback(UART_HandleTypeDef *UartHandle) {
    BSP_LED_On(LED3);
}

/* System Clock (180MHz) */
static void SystemClock_Config(void)
{
  RCC_ClkInitTypeDef RCC_ClkInitStruct;
  RCC_OscInitTypeDef RCC_OscInitStruct;

  /* Enable Power Control clock */
  __HAL_RCC_PWR_CLK_ENABLE();

  /* The voltage scaling allows optimizing the power consumption when the device is
     clocked below the maximum system frequency, to update the voltage scaling value
     regarding system frequency refer to product datasheet.  */
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /* Enable HSE Oscillator and activate PLL with HSE as source */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 8;
  RCC_OscInitStruct.PLL.PLLN = 360;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 7;
  if(HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /* Activate the Over-Drive mode */
  HAL_PWREx_EnableOverDrive();

  /* Select PLL as system clock source and configure the HCLK, PCLK1 and PCLK2
     clocks dividers */
  RCC_ClkInitStruct.ClockType = (RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2);
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;
  if(HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK)
  {
    Error_Handler();
  }
}

static void Error_Handler(void) {
    BSP_LED_On(LED4);
    while(1);
}
