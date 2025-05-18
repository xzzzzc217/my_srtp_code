//用于heltec wifi lora32 v3(红盒子)
#include "LoRaWan_APP.h"
#include "Arduino.h"

// 设置通信频率为 433 MHz。
#define RF_FREQUENCY                                433000000 // Hz

// 设置发送功率为 17 dBm。
#define TX_OUTPUT_POWER                             17        // dBm

// 设置带宽为 125 kHz。
#define LORA_BANDWIDTH                              0         // [0: 125 kHz,
//  1: 250 kHz,
//  2: 500 kHz,
//  3: Reserved]

// 设置扩频因子为 SF7（适合低功耗长距离传输）。
#define LORA_SPREADING_FACTOR                       7         // [SF7..SF12]

// 设置编码率为 4/5。
#define LORA_CODINGRATE                             1         // [1: 4/5,
//  2: 4/6,
//  3: 4/7,
//  4: 4/8]

// 设置前导码长度。
#define LORA_PREAMBLE_LENGTH                        8         // Same for Tx and Rx

#define LORA_SYMBOL_TIMEOUT                         0         // Symbols

// 是否启用固定长度数据包。
#define LORA_FIX_LENGTH_PAYLOAD_ON                  false     

#define LORA_IQ_INVERSION_ON                        false

#define RX_TIMEOUT_VALUE                            1000

// 定义数据包缓冲区大小。
#define BUFFER_SIZE                                 30 

// 定义两组双帧之间的间隔时间（毫秒）。
#define FRAME_GROUP_INTERVAL                        300 // 0.3秒

char txpacket[BUFFER_SIZE];
char rxpacket[BUFFER_SIZE];

double txNumber;
int counter = 0;
bool lora_idle = true;

static RadioEvents_t RadioEvents;

// 发送完成回调函数
void OnTxDone(void);
// 发送超时回调函数
void OnTxTimeout(void);

void setup() {
    Serial.begin(115200);
    Mcu.begin(HELTEC_WIFI_LORA_32_V3, 0);

    txNumber = 0;

    RadioEvents.TxDone = OnTxDone;
    RadioEvents.TxTimeout = OnTxTimeout;

    Radio.Init(&RadioEvents);
    Radio.SetChannel(RF_FREQUENCY);
    Radio.SetTxConfig(MODEM_LORA, TX_OUTPUT_POWER, 0, LORA_BANDWIDTH,
                      LORA_SPREADING_FACTOR, LORA_CODINGRATE,
                      LORA_PREAMBLE_LENGTH, LORA_FIX_LENGTH_PAYLOAD_ON,
                      true, 0, 0, LORA_IQ_INVERSION_ON, 3000);
}

void loop() {
    if (lora_idle == true) {
        // 发送第一帧
        sprintf(txpacket, "czc-lora-first-%d", counter);
        Serial.printf("\r\nSending frame packet 1: \"%s\", length %d, count: %d, send time: %lu\r\n",
                      txpacket, strlen(txpacket), counter, millis());
        Radio.Send((uint8_t *)txpacket, strlen(txpacket));
        lora_idle = false;
        
        // 等待第一帧发送完成
        while (lora_idle == false) {
            Radio.IrqProcess();
        }

        // 发送第二帧
        sprintf(txpacket, "czc-lora-second-%d", counter);
        Serial.printf("\r\nSending frame packet 2: \"%s\", length %d, count: %d, send time: %lu\r\n",
                      txpacket, strlen(txpacket), counter, millis());
        Radio.Send((uint8_t *)txpacket, strlen(txpacket));
        lora_idle = false;
        
        // 等待第二帧发送完成
        while (lora_idle == false) {
            Radio.IrqProcess();
        }

        counter++; // 每发送一对帧后递增计数器

        // 在发送下一组双帧前等待指定间隔
        Serial.printf("Waiting for %d ms before next frame group...\r\n", FRAME_GROUP_INTERVAL);
        delay(FRAME_GROUP_INTERVAL);
    }
}

// 当 LoRa 模块完成数据包的发送后，OnTxDone 函数被调用，
// 输出 "TX done......"，并将 lora_idle 设置为 true，表示可以重新发送。
void OnTxDone(void) {
    //Serial.println("TX done......");对实验可能有影响，因此注释掉
    lora_idle = true;
}

// 如果发送过程中超时，则会调用 OnTxTimeout 函数，
// 输出 "TX Timeout......"，并将模块置于睡眠模式，进入空闲状态。从未触发过
void OnTxTimeout(void) {
    Radio.Sleep();
    //Serial.println("TX Timeout......");对实验可能有影响，因此注释掉
    lora_idle = true;
}