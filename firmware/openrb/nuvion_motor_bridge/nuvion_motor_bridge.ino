#include <Dynamixel2Arduino.h>
#include "position_reference.h"
using namespace ControlTableItem;
Dynamixel2Arduino dxl(Serial1, -1);
struct Motor { uint8_t id; bool present; uint16_t model; int32_t position, goal, current, mode, torque, hw; int32_t homingOffset; bool referenceValid; };
Motor motors[2]={{1,false,0,0,0,0,0,0,0,0,false},{2,false,0,0,0,0,0,0,0,0,false}};
struct Found { uint8_t id; uint16_t model; uint32_t baud; };
Found foundMotors[32]; uint8_t foundCount=0;
bool armed=false, running=false;
uint8_t runningId=0; uint32_t lastRun=0;
uint32_t lastControl=0, lastMonitor=0, busBaud=57600;
const char* fault="";
struct Input { char line[80]; uint8_t used=0; bool overflow=false; } usbIn, uart2In, uart3In;

bool readItem(uint8_t item,uint8_t id,int32_t &out) {
  out=dxl.readControlTableItem(item,id,20);
  return dxl.getLastLibErrCode()==0 && dxl.getLastStatusPacketError()==0;
}
bool writeItem(uint8_t item,uint8_t id,int32_t value) {
  return dxl.writeControlTableItem(item,id,value,20) && dxl.getLastStatusPacketError()==0;
}
bool refresh(Motor &m) {
  if(!m.present)return false;
  int32_t p,c,t,h,o;
  if(!readItem(PRESENT_POSITION,m.id,p)||!readItem(PRESENT_CURRENT,m.id,c)||!readItem(TORQUE_ENABLE,m.id,t)||!readItem(HARDWARE_ERROR_STATUS,m.id,h)||!readItem(HOMING_OFFSET,m.id,o))return false;
  m.position=p;m.current=(int16_t)c;m.torque=t;m.hw=h;m.homingOffset=o;
  int32_t referenced; m.referenceValid=controlPosition(p,m.mode,t,o,referenced);return true;
}
bool off() {
  running=false;runningId=0;
  bool ok=true;
  for(auto &m:motors) if(m.present) { if(!writeItem(TORQUE_ENABLE,m.id,0))ok=false; else m.torque=0; }
  armed=false;return ok;
}
void discover() {
  armed=false;foundCount=0;
  for(auto &m:motors)m.present=false;
  const uint32_t rates[]={57600,1000000,115200,9600,2000000,3000000,4000000};
  uint32_t chosen=0;
  for(auto baud:rates) {
    dxl.begin(baud);delay(50);
    DYNAMIXEL::InfoFromPing_t infos[16];
    uint8_t n=static_cast<DYNAMIXEL::Master&>(dxl).ping(254,infos,16,350);
    for(uint8_t i=0;i<n;i++) {
      if(foundCount<32)foundMotors[foundCount++]={infos[i].id,infos[i].model_number,baud};
      if(!chosen)chosen=baud;
    }
  }
  busBaud=chosen?chosen:57600;dxl.begin(busBaud);delay(50);
  for(auto &m:motors) {
    m.present=dxl.ping(m.id);m.model=m.present?dxl.getModelNumber(m.id):0;
    if(m.present){readItem(OPERATING_MODE,m.id,m.mode);refresh(m);m.goal=m.position;}
  }
  off();
}
bool arm() {
  if(armed){fault="already_armed";return false;}
  for(auto &m:motors) {
    if(!m.present || (m.model!=1190 && m.model!=1200)){fault="need_two_XL330_ids_1_2";return false;}
    if(!readItem(OPERATING_MODE,m.id,m.mode)||(m.mode!=3&&m.mode!=5)||!refresh(m)||m.hw||!m.referenceValid){fault="motor_mode_or_position_or_error";return false;}
    int32_t target,low,high;
    if(!controlPosition(m.position,m.mode,m.torque,m.homingOffset,target)
       ||!readItem(MIN_POSITION_LIMIT,m.id,low)||!readItem(MAX_POSITION_LIMIT,m.id,high)
       ||target<low||target>high){fault="position_reference_outside_limits";return false;}
    int32_t drive;
    if(!readItem(DRIVE_MODE,m.id,drive)||(drive&4)){fault="time_profile_not_supported";return false;}
  }
  for(auto &m:motors) {
    int32_t limit;
    if(!writeItem(TORQUE_ENABLE,m.id,0)||!writeItem(BUS_WATCHDOG,m.id,0)||!readItem(PWM_LIMIT,m.id,limit)||!writeItem(GOAL_PWM,m.id,min(limit,(int32_t)180))||!writeItem(PROFILE_ACCELERATION,m.id,5)||!writeItem(PROFILE_VELOCITY,m.id,10)){off();fault="limit_setup_failed";return false;}
    if(m.mode==5) { if(!readItem(CURRENT_LIMIT,m.id,limit)||!writeItem(GOAL_CURRENT,m.id,min(limit,(int32_t)200))){off();fault="current_setup_failed";return false;} }
    if(!refresh(m)){off();fault="position_read_failed";return false;}
    int32_t target,low,high;
    if(!controlPosition(m.position,m.mode,m.torque,m.homingOffset,target)
       ||!readItem(MIN_POSITION_LIMIT,m.id,low)||!readItem(MAX_POSITION_LIMIT,m.id,high)
       ||target<low||target>high){off();fault="position_reference_outside_limits";return false;}
    // Write the post-enable coordinate BEFORE enabling torque. Never enable
    // against the old goal, which may be nearly a whole turn away.
    m.goal=target;
    if(!writeItem(GOAL_POSITION,m.id,target)||!writeItem(TORQUE_ENABLE,m.id,1)){
      off();fault="arm_failed";return false;
    }
    // Read back the reset encoder and verify no unexpected reference change.
    if(!refresh(m)||!m.torque||m.hw||m.position<low||m.position>high
       ||abs(m.position-target)>23){off();fault="position_reference_changed";return false;}
    m.goal=m.position;
    if(!writeItem(GOAL_POSITION,m.id,m.goal)||!writeItem(BUS_WATCHDOG,m.id,50)){
      off();fault="arm_failed";return false;
    }
  }
  armed=true;lastControl=millis();fault="";return true;
}
bool jog(int id,int step) {
  if(running){fault="stop_before_step";return false;}
  if(!armed || id<1 || id>2 || (step!=-1&&step!=1)){fault="not_armed_or_bad_step";return false;}
  Motor &m=motors[id-1];
  if(!refresh(m)){off();fault="motor_read_failed";return false;}
  int32_t goal=constrain(m.goal+step*11,(int32_t)0,(int32_t)4095);
  if(goal==m.goal){fault="position_limit";return false;}
  if(abs(m.position-m.goal)>23){fault="waiting_for_motion_or_motor_stalled";return false;}
  if(!writeItem(GOAL_POSITION,m.id,goal)){off();fault="goal_write_failed";return false;}
  m.goal=goal;lastControl=millis();fault="";return true;
}
bool run(int id,int direction,int speed) {
  if(!armed || running || id<1 || id>2 || (direction!=-1&&direction!=1) || (speed!=2&&speed!=4&&speed!=8)){fault="invalid_run";return false;}
  Motor &m=motors[id-1];
  if(m.mode!=3 || !refresh(m) || m.hw || !m.torque){off();fault="motor_not_ready";return false;}
  int32_t low,high;
  if(!readItem(MIN_POSITION_LIMIT,m.id,low)||!readItem(MAX_POSITION_LIMIT,m.id,high)||low<0||high>4095||low>=high){off();fault="position_limits_read_failed";return false;}
  int32_t goal=direction>0?high:low;
  if(abs(goal-m.position)<=2){fault="position_limit";return false;}
  if(!writeItem(PROFILE_VELOCITY,m.id,speed)||!writeItem(GOAL_POSITION,m.id,goal)){off();fault="run_write_failed";return false;}
  m.goal=goal;running=true;runningId=m.id;lastRun=lastControl=millis();fault="";return true;
}
bool hold() {
  running=false;runningId=0;
  if(!armed)return true;
  for(auto &m:motors) {
    if(!refresh(m)||!writeItem(GOAL_POSITION,m.id,m.position)){off();fault="hold_failed";return false;}
    m.goal=m.position;
  }
  lastControl=millis();return true;
}
void reply(Stream &s,long seq,bool ok,const char* link) {
  s.print("{\"protocol\":\"NUV1\",\"seq\":");s.print(seq);
  s.print(",\"ok\":");s.print(ok?"true":"false");
  s.print(",\"firmware\":\"NUV1-smooth-v4\"");
  s.print(",\"moving_id\":");s.print(runningId);
  s.print(",\"armed\":");s.print(armed?"true":"false");
  s.print(",\"link\":\"");s.print(link);s.print("\",\"baud\":");s.print(busBaud);
  s.print(",\"error\":\"");s.print(fault);s.print("\",\"motors\":[");
  for(int i=0;i<2;i++) {
    auto &m=motors[i];if(i)s.print(',');s.print("{\"id\":");s.print(m.id);
    s.print(",\"present\":");s.print(m.present?"true":"false");
    s.print(",\"model\":");s.print(m.model);s.print(",\"position\":");
    int32_t referenced=m.position;
    if(m.referenceValid)controlPosition(m.position,m.mode,m.torque,m.homingOffset,referenced);
    s.print(referenced);
    s.print(",\"raw_position\":");s.print(m.position);
    s.print(",\"reference_valid\":");s.print(m.referenceValid?"true":"false");
    s.print(",\"goal\":");s.print(m.goal);s.print(",\"current\":");s.print(m.current);
    s.print(",\"torque\":");s.print(m.torque);s.print(",\"mode\":");s.print(m.mode);
    s.print(",\"hardware_error\":");s.print(m.hw);s.print('}');
  }
  s.print("],\"found\":[");
  for(uint8_t i=0;i<foundCount;i++){if(i)s.print(',');s.print("{\"id\":");s.print(foundMotors[i].id);s.print(",\"model\":");s.print(foundMotors[i].model);s.print(",\"baud\":");s.print(foundMotors[i].baud);s.print('}');}
  s.println("]}");
}
void command(Stream &s,char *line,const char *link) {
  long seq;char cmd[16];int id=0,step=0,speed=0;
  if(sscanf(line,"NUV1 %ld %15s %d %d %d",&seq,cmd,&id,&step,&speed)<2)return;
  bool ok=true;
  if(!strcmp(cmd,"STATUS")) {for(auto &m:motors)if(m.present&&!refresh(m)){off();fault="motor_read_failed";ok=false;}}
  else if(!strcmp(cmd,"HEARTBEAT")){lastControl=millis();}
  else if(!strcmp(cmd,"ARM"))ok=arm();
  else if(!strcmp(cmd,"RUN"))ok=run(id,step,speed);
  else if(!strcmp(cmd,"KEEP")){if(armed&&running)lastRun=lastControl=millis();else{fault="run_not_active";ok=false;}}
  else if(!strcmp(cmd,"JOG"))ok=jog(id,step);
  else if(!strcmp(cmd,"STOP"))ok=hold();
  else if(!strcmp(cmd,"OFF")){ok=off();fault=ok?"":"torque_off_failed";}
  else if(!strcmp(cmd,"SCAN")&&!armed){discover();fault="";}
  else {fault="unknown_command";ok=false;}
  reply(s,seq,ok,link);
}
void consume(Stream &s,Input &in,const char *link) {
  while(s.available()) {
    char c=s.read();
    if(c=='\n') {if(!in.overflow){in.line[in.used]=0;command(s,in.line,link);}in.used=0;in.overflow=false;}
    else if(c!='\r') {if(in.used<sizeof(in.line)-1)in.line[in.used++]=c;else in.overflow=true;}
  }
}
void setup() {
  Serial.begin(115200);Serial2.begin(115200);Serial3.begin(115200);
  pinMode(BDPIN_DXL_PWR_EN,OUTPUT);digitalWrite(BDPIN_DXL_PWR_EN,HIGH);
  dxl.setPortProtocolVersion(2.0);discover();
}
void loop() {
  if(armed&&running&&millis()-lastRun>400){if(hold())fault="motion_timeout";}
  consume(Serial,usbIn,"USB");consume(Serial2,uart2In,"Serial2");consume(Serial3,uart3In,"Serial3");
  if(armed&&millis()-lastControl>2000){off();fault="control_timeout";}
  if(armed&&millis()-lastMonitor>100){lastMonitor=millis();for(auto &m:motors) {
    if(!refresh(m)||m.hw||abs(m.current)>350){off();fault="motor_fault_or_overcurrent";break;}
  }}
}
