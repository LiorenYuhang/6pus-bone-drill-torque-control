"""Targeted IMU/reference and old-run regression; never open motor serial."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from imu_chain import StationaryImuReference, ImuToLengths
from calibration import validate_relative_start_reference
from uart_driver import SixAxisPositionDrive
from pus6_motor_driver.fake_bus import FakeBus

ROOT=Path(__file__).resolve().parents[4]


def test_stationary_average_preserves_real_nonzero_heading_and_quaternion_sign():
    capture=StationaryImuReference();q=Rotation.from_euler('xyz',[.1,-.2,1.]).as_quat()
    for i in range(111):
        for role in ('base','platform'):
            capture.add(role,q*(-1 if i%2 else 1),[.001,0,0],1_000_000_000+i*10_000_000)
    result=capture.capture()
    assert result is not None
    assert abs(np.dot(result['base']['q_xyzw'],q))==pytest.approx(1.)
    assert result['base']['gyro_mean_rad_s']==pytest.approx([.001,0,0])
    assert result['base']['samples']>=50


def test_stationary_reference_needs_unique_data_and_real_available_gyro():
    capture=StationaryImuReference()
    for i in range(100):
        capture.add('base',[0,0,0,1],[0,0,0],1_000_000_000)
        capture.add('platform',[0,0,0,1],None,1_000_000_000+i*20_000_000)
    assert capture.capture() is None


def test_moving_reference_is_not_blindly_reset_to_zero():
    capture=StationaryImuReference()
    for i in range(101):
        for role in ('base','platform'):
            capture.add(role,Rotation.from_euler('x',i*.002).as_quat(),[.2,0,0],1_000_000_000+i*10_000_000)
    assert capture.capture() is None and 'moving' in capture.reason


def test_j14_mount_and_so3_remain_unchanged_no_gyro_integration():
    from j11_base_feedforward_runner import relative_quat, apply_z90_transform, quat_to_rot_matrix
    from j11_m2_pid_residual_runner import compute_so3_platform_error,compose_so3_pid_residual
    c=ImuToLengths(base_reference=[0,0,0,1],platform_reference=[0,0,0,1],
        kp=[.5]*3,ki=[0]*3,kd=[0]*3,residual_limit_rad=[.02]*3,
        integral_limit_rad_s=[.02]*3,shaper={})
    base=Rotation.from_euler('xyz',[.01,-.02,.03]).as_quat()
    plat=Rotation.from_euler('xyz',[-.02,.01,-.01]).as_quat()
    c.step(base,plat,1_000_000_000);result=c.step(base,plat,1_020_000_000)
    rb=quat_to_rot_matrix(apply_z90_transform(relative_quat(c.base0,base)))
    rp=quat_to_rot_matrix(apply_z90_transform(relative_quat(c.platform0,plat)))
    _,err=compute_so3_platform_error(np.eye(3),rp)
    _,_,command,_=compose_so3_pid_residual(rb,rb.T,np.clip(.5*err,-.02,.02))
    assert result['pose_m_rad'][3:]==pytest.approx(Rotation.from_matrix(command).as_euler('xyz'))


def test_recorded_old_force_reference_rejected_before_any_position_or_mode_write():
    old=json.loads((ROOT/'results/force_position/20261105-125125/config.json').read_text())
    rows=[json.loads(s) for s in (ROOT/'results/force_position/20261105-125125/telemetry.jsonl').read_text().splitlines()]
    first=next(r['data'] for r in rows if r['kind']=='feedback')
    with pytest.raises(ValueError,match='relative force reference'):
        validate_relative_start_reference(old,first['force_raw'])
    bus=FakeBus();bus.modes={i:0 for i in range(1,7)}
    drive=SixAxisPositionDrive(bus,max_start_delta_steps=None,max_cycle_delta_steps=None,event=lambda *_:None)
    with pytest.raises(ValueError,match='relative force reference'):
        drive.initialize(active=True,select_servo=True,
            initial_feedback_validator=lambda _:validate_relative_start_reference(old,first['force_raw']))
    assert bus.write_audit==[]


def test_relative_reference_check_is_signed_not_automatic_tare():
    config=dict(calibration=dict(kind='relative_tare',raw_zero=[100]*6,relative_scale=[-1]*6),
                controller=dict(force_limit_relative=[50]*6))
    result=validate_relative_start_reference(config,[110]*6)
    assert result['initial_offset_relative']==[-10]*6
    assert config['calibration']['raw_zero']==[100]*6
    with pytest.raises(ValueError):validate_relative_start_reference(config,[1000]*6)
    assert validate_relative_start_reference({'calibration':{'newtons_per_raw':[.01]*6}},[1000]*6) is None
