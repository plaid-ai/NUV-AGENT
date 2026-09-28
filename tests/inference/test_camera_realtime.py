from unittest.mock import Mock, patch
import time

from nuvion_app.inference.camera_realtime import CameraRealtimeControl
from nuvion_app.inference.motor import MotorLimitReached


def setup():
    motor = Mock()
    response = {'motors': [{'id': 1, 'present': True, 'position': 2000}, {'id': 2, 'present': True, 'position': 1200}]}
    motor.position_action.return_value = response
    motor.realtime_step.return_value = response
    motor.position_limits = dict(panMin=0, panMax=4095, tiltMin=0, tiltMax=4095)
    publish = Mock()
    return CameraRealtimeControl(motor, publish, start=False), motor, publish


def frame(action='LEFT', sequence=1, session='owner'):
    return dict(action=action, sequence=sequence, sessionId=session, requestId='request', expiresAtMs=int(time.time()*1000)+600)


import unittest


class CameraRealtimeTest(unittest.TestCase):
    def test_move_reports_measured_angle_without_waiting_for_settle(self):
        control, motor, publish = setup()
        assert control.accept(frame())
        control.tick()
        motor.realtime_step.assert_called_once()
        motor.move_position.assert_not_called()
        assert publish.call_args.args[0]['status'] == 'MOVING'
        assert publish.call_args.args[0]['telemetry']['positions']['pan'] == 2000


    def test_release_stops_without_waiting_for_previous_ack(self):
        control, motor, publish = setup()
        control.accept(frame()); control.tick()
        control.accept(frame('STOP', 2)); control.tick()
        motor.position_action.assert_called_with('STOP')
        assert publish.call_args.args[0]['status'] == 'STOPPED'
        assert not control.accept(frame(sequence=1))
        assert motor.realtime_step.call_count == 1


    def test_deadman_uses_original_expiry_not_time_of_receipt(self):
        control, motor, publish = setup()
        request = frame()
        control.accept(request); control.tick()
        with patch('nuvion_app.inference.camera_realtime.time.time', return_value=request['expiresAtMs']/1000+0.01):
            control.tick()
        motor.position_action.assert_called_with('STOP')
        assert motor.realtime_step.call_count == 1


    def test_expired_or_future_input_cannot_move(self):
        control, motor, _ = setup()
        for expiry in (int(time.time()*1000)-1, int(time.time()*1000)+2000):
            assert not control.accept({**frame(), 'expiresAtMs': expiry})
        control.tick()
        motor.realtime_step.assert_not_called()


    def test_disconnect_stops_and_never_resumes(self):
        control, motor, _ = setup()
        control.accept(frame()); control.tick(); control.disconnect(); control.tick(); control.tick()
        assert motor.realtime_step.call_count == 1
        motor.position_action.assert_called_with('STOP')


    def test_other_viewer_cannot_take_over_active_press_but_can_stop(self):
        control, motor, publish = setup()
        control.accept(frame()); control.tick()
        assert not control.accept(frame(session='other'))
        assert publish.call_args.args[0]['status'] == 'BUSY'
        assert control.accept(frame('STOP', 2, 'other'))
        control.tick()
        motor.position_action.assert_called_with('STOP')


    def test_limit_keeps_angles_and_stops_without_retry(self):
        control, motor, publish = setup()
        motor.realtime_step.side_effect = MotorLimitReached('limit')
        control.accept(frame()); control.tick(); control.tick()
        assert motor.realtime_step.call_count == 1
        result = publish.call_args.args[0]
        assert result['status'] == 'LIMIT_REACHED'
        assert result['telemetry']['positions']['pan'] == 2000


    def test_transport_generation_invalidates_pending_movement(self):
        control, motor, _ = setup()
        control.accept(frame()); control.disconnect(); control.tick()
        motor.realtime_step.assert_not_called()
        motor.position_action.assert_called_with('STOP')

    def test_fault_requires_stop_before_renewed_motion(self):
        control, motor, _ = setup()
        motor.realtime_step.side_effect = MotorLimitReached('limit')
        control.accept(frame()); control.tick()
        assert not control.accept(frame(sequence=2))
        assert control.accept(frame('STOP', 3)); control.tick()
        motor.realtime_step.side_effect = None
        assert control.accept(frame(sequence=4)); control.tick()
        assert motor.realtime_step.call_count == 2

    def test_idle_status_does_not_take_automatic_tracking_lease(self):
        control, motor, _ = setup()
        motor.manual_control_until = 0
        control.accept(frame('STATUS')); control.tick()
        assert motor.manual_control_until == 0

    def test_another_owners_stop_fences_previous_held_key(self):
        control, motor, _ = setup()
        control.accept(frame()); control.tick()
        control.accept(frame('STOP', 1, 'other')); control.tick()
        assert not control.accept(frame(sequence=2))
        assert motor.realtime_step.call_count == 1
        assert control.accept(frame('STOP', 3))
        assert control.accept(frame(sequence=4))
