#pragma once
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <vector>
using std::min;
inline int32_t constrain(int32_t v,int32_t lo,int32_t hi){return std::max(lo,std::min(v,hi));}
namespace ControlTableItem { enum Item { PRESENT_POSITION, PRESENT_CURRENT, TORQUE_ENABLE, HARDWARE_ERROR_STATUS, HOMING_OFFSET, OPERATING_MODE, DRIVE_MODE, MIN_POSITION_LIMIT, MAX_POSITION_LIMIT, BUS_WATCHDOG, PWM_LIMIT, GOAL_PWM, PROFILE_ACCELERATION, PROFILE_VELOCITY, CURRENT_LIMIT, GOAL_CURRENT, GOAL_POSITION }; }
class Stream { public: template<class T>void print(T){} template<class T>void println(T){} void begin(int){} bool available(){return false;} char read(){return 0;} };
static Stream Serial,Serial1,Serial2,Serial3;
const int BDPIN_DXL_PWR_EN=0,OUTPUT=0,HIGH=1;
inline void pinMode(int,int){} inline void digitalWrite(int,int){} inline void delay(int){} inline uint32_t millis(){return 100;}
namespace DYNAMIXEL { struct InfoFromPing_t {uint8_t id;uint16_t model_number;}; class Master { public:uint8_t ping(int,InfoFromPing_t*,int,int){return 0;} }; }
struct Write {uint8_t item,id;int32_t value;};
static int32_t registers[3][32];
static std::vector<Write> writes;
static int enableDrift=0;
class Dynamixel2Arduino:public DYNAMIXEL::Master { public:
 Dynamixel2Arduino(Stream&,int){} void begin(uint32_t){} void setPortProtocolVersion(int){}
 bool ping(uint8_t){return true;} uint16_t getModelNumber(uint8_t){return 1200;}
 int getLastLibErrCode(){return 0;} int getLastStatusPacketError(){return 0;}
 int32_t readControlTableItem(uint8_t item,uint8_t id,int){return registers[id][item];}
 bool writeControlTableItem(uint8_t item,uint8_t id,int32_t value,int){
  writes.push_back({item,id,value});registers[id][item]=value;
  if(item==ControlTableItem::TORQUE_ENABLE && value==1){
   auto &p=registers[id][ControlTableItem::PRESENT_POSITION];p=(p%4096+4096)%4096+enableDrift;
  }
  return true;
 }
};
