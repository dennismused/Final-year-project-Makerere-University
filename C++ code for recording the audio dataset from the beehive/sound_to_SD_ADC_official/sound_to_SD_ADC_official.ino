#include "FS.h"
#include "SD.h"
#include "SPI.h"
#include "Wire.h"

#define SD_CS 5
#define MIC_PIN_1 35
#define VIB_PIN_2 34
#define LED_BUILTIN 2

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    ;
  } 

  if (!SD.begin(SD_CS)) {
    Serial.println("Card Mount Failed");
    return;
  }

  Serial.println("Card Mount Success");

  pinMode(MIC_PIN_1, INPUT);
  pinMode(VIB_PIN_2, INPUT);
  pinMode(LED_BUILTIN, OUTPUT);  

  Serial.println("Setup done");

}

void loop() {

  File file = SD.open("/recording/REC.wav", FILE_APPEND); 

  if (!file ) {
    Serial.println("Failed to open file in writing mode");
    return;
  }

  int16_t sample1;
  int16_t sample2;
    
  Serial.println("Recording...");
  digitalWrite(LED_BUILTIN, HIGH);

  for (int i = 0; i < 600000/6; i++) {
    sample1 = analogRead(MIC_PIN_1);
    sample2 = analogRead(VIB_PIN_2);
    file.write((uint8_t *)&sample1, sizeof(sample1));
    file.write((uint8_t *)&sample2, sizeof(sample2));
  }

  file.close();

  Serial.println("Recording done");

}
