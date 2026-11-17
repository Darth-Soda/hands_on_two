#include <Wire.h>
#include "kxtj3-1057.h"

#define SDApin 24
#define SCLpin 25


float sampleRate = 100; //using high sample rate
uint8_t accelRange = 16; 
bool highRes = true;

KXTJ3 accelerometer(0x0E);
unsigned long sampleIntervalMicros;
unsigned long lastSampleMicros = 0;

void setup() {
  Serial.begin(115200);
  delay(1000);

  Wire.setSDA(SDApin);
  Wire.setSCL(SCLpin);
  Wire.begin();

if(accelerometer.begin(sampleRate,accelRange, highRes) == IMU_SUCCESS){ 
  Serial.println("IMU init."); }

else{Serial.println(" init. failed"); while(true);}
  sampleIntervalMicros = (unsigned long)(1e6 / sampleRate); //Number of microseconds between each sample rate
  lastSampleMicros = micros(); //no of microseconds since boot

}

void loop() {
  unsigned long now = micros();

  if (now - lastSampleMicros >= sampleIntervalMicros) { //if no. of microseconds elapsed since last sample> time b/w each sample
    lastSampleMicros = now;

    float ax = accelerometer.axisAccel(X);
    float ay = accelerometer.axisAccel(Y);
    float az = accelerometer.axisAccel(Z);

    Serial.print(now/1e6); //time in seconds
    Serial.print(',');
    Serial.print(ax, 4);
    Serial.print(',');
    Serial.print(ay, 4);
    Serial.print(',');
    Serial.println(az, 4);
  }
}

