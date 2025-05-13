#include <SPI.h>
#include <LoRa.h>

// 定义两组双帧之间的间隔时间（毫秒）
#define FRAME_GROUP_INTERVAL 1000 // 5秒

int counter = 0;

void setup() {
  Serial.begin(9600);
  while (!Serial);
  Serial.println("LoRa Sender");

  // 初始化 LoRa 模块，频率为 433 MHz
  if (!LoRa.begin(433E6)) {
    Serial.println("Starting LoRa failed!");
    while (1);
  }

  // 配置 LoRa 参数（可选，优化传输）
  LoRa.setTxPower(17); // 设置发送功率为 17 dBm
  LoRa.setSpreadingFactor(7); // 设置扩频因子为 SF7
  LoRa.setSignalBandwidth(125E3); // 设置带宽为 125 kHz
  LoRa.setCodingRate4(5); // 设置编码率为 4/5
  Serial.println("LoRa initialized successfully");
}

void loop() {
  // 发送第一帧
  Serial.print("Sending packet 1: ");
  Serial.println(counter);
  LoRa.beginPacket();
  LoRa.print("hello-first-");
  LoRa.print(counter);
  LoRa.endPacket();
  
  // 发送第二帧
  Serial.print("Sending packet 2: ");
  Serial.println(counter);
  LoRa.beginPacket();
  LoRa.print("hello-second-");
  LoRa.print(counter);
  LoRa.endPacket();

  counter++; // 每发送一组双帧后递增计数器

  // 在发送下一组双帧前等待指定间隔
  Serial.print("Waiting for ");
  Serial.print(FRAME_GROUP_INTERVAL);
  Serial.println(" ms before next frame group...");
  delay(FRAME_GROUP_INTERVAL);
}