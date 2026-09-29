#include <cassert>
#include "../nuvion_motor_bridge/nuvion_motor_bridge.ino"
void reset(int32_t tilt=4395){
 writes.clear();memset(registers,0,sizeof(registers));armed=false;enableDrift=0;
 for(int id=1;id<=2;id++){
  motors[id-1]={static_cast<uint8_t>(id),true,1200,0,0,0,3,0,0,0,false};
  registers[id][OPERATING_MODE]=3;registers[id][PRESENT_POSITION]=id==1?2928:tilt;
  registers[id][MAX_POSITION_LIMIT]=4095;registers[id][PWM_LIMIT]=885;
  registers[id][GOAL_POSITION]=4051;
 }
}
int main(){
 for(int32_t raw:{4395,-1,8192}){
  reset(raw);assert(arm());assert(armed);
  int32_t lastGoal[3]={-1,-1,-1};int enabled=0;
  for(auto w:writes){
   if(w.item==GOAL_POSITION)lastGoal[w.id]=w.value;
   if(w.item==TORQUE_ENABLE&&w.value==1){
    assert(lastGoal[w.id]==(w.id==1?2928:(raw%4096+4096)%4096));enabled++;
   }
  }
  assert(enabled==2);assert(motors[1].position==(raw%4096+4096)%4096);
 }
 reset();registers[2][HOMING_OFFSET]=10;assert(!arm());assert(writes.empty());
 reset();registers[2][OPERATING_MODE]=5;assert(!arm());assert(writes.empty());
 reset();registers[2][MIN_POSITION_LIMIT]=1000;assert(!arm());assert(writes.empty());
 reset();enableDrift=100;assert(!arm());assert(!armed);
 assert(registers[1][TORQUE_ENABLE]==0&&registers[2][TORQUE_ENABLE]==0);
}
