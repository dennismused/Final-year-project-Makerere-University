#include "FS.h"
#include "SD.h"
#include "SPI.h"
#include "Wire.h"

#include <WiFi.h>
#include <WiFiClient.h>
#include <WebServer.h>
#include <ESPmDNS.h>
#include <HardwareSerial.h> //For GSM

HardwareSerial SIM800C(1);  //For GSM

#define SD_CS 5
#define MIC_PIN_1 35
#define VIB_PIN_2 34
#define LED_BUILTIN 2


// Replace with your network credentials
const char* ssid = "ESP32"; //The SSID of your network
const char* password = "123456789"; //The password of your network

// Web server
WebServer server(80);

// SD card
const int chipSelect = 5; //The CS pin

// Audio file name
const char* audioFile = "/recording/REC.wav";

// MIME type for audio file
const char* audioType = "application/octet-stream"; //To indicate binary data

// Function to handle root path
void handleRoot() {
  // Send a simple HTML page
  server.send(200, "text/html", "<h1>ESP32 SD Web Server</h1><p><a href=\"/audio\">Download audio</a></p>");
}

// Function to handle audio path
void handleAudio() {
  
  //First record the audio    
  File file = SD.open("/recording/REC.wav", FILE_WRITE); 

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
  digitalWrite(LED_BUILTIN, LOW);
    
  // Check if the file exists
  if (SD.exists(audioFile)) {
    // Open the file for reading
    File file = SD.open(audioFile);
    if (file) {
      // Send the file with the appropriate MIME type and content disposition header
      server.sendHeader("Content-Disposition", "attachment; filename=audio.wav"); //To suggest a filename and trigger download
      server.streamFile(file, audioType);
      // Close the file
      file.close();
    }
    else {
      // Send an error message
      server.send(500, "text/plain", "Failed to open file");
    }
  }
  else {
    // Send a not found message
    server.send(404, "text/plain", "File not found");
  }
}

// Function to make a phone call
void phoneCall() {

  SIM800C.begin(115200, SERIAL_8N1, 4, 2);  //TXD(4), RXD(2)
  delay(1000);
  SIM800C.print("ATD+256703203625;\r");
  server.send(200);
  
}

// Function to setup WiFi as an access point
void setupWiFi() {

  Serial.print("Setting ESP32 Access Point…");
  WiFi.softAP(ssid, password); 
  IPAddress IP = WiFi.softAPIP();
  Serial.println("");
  Serial.print("ESP32 IP address: ");
  Serial.println(IP);
}

// Function to setup MDNS
void setupMDNS() {
  
  // Start MDNS responder
  if (MDNS.begin("esp32")) {
    Serial.println("MDNS responder started");
    // Add service to MDNS-SD
    MDNS.addService("http", "tcp", 80);
    
  }
}

// Function to setup web server
void setupServer() {
  
  // Handle root path
  server.on("/", handleRoot);
 
  // Handle audio path
  server.on("/audio", handleAudio);

  // Handle phone call path
  server.on("/phonecall", phoneCall);
  
  // Start web server
  server.begin();
  
  Serial.println("Web server started");
}

// Function to setup SD card
void setupSD() {
  
  // Initialize SPI bus
  SPI.begin();
 
  // Initialize SD card
  if (!SD.begin(chipSelect)) {
    Serial.println("Card Mount Failed");
    return;
  }
  
  uint8_t cardType = SD.cardType();

  if (cardType == CARD_NONE) {
  Serial.println("No SD card attached");
  return;
  }

  Serial.println("SD card initialized");
}


void setup() {

  Serial.begin(115200);

  while (!Serial) {
  ;
  }

  pinMode(MIC_PIN_1, INPUT);
  pinMode(VIB_PIN_2, INPUT);
  pinMode(LED_BUILTIN, OUTPUT);

  // Setup WiFi as an access point
  setupWiFi();

  // Setup MDNS
  setupMDNS();

  // Setup web server
  setupServer();

  // Setup SD card
  setupSD();

  Serial.println("Setup done");
}

void loop() {

  // Handle web server requests 
  server.handleClient();

}
