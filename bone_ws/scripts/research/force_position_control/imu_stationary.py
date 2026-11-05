#!/usr/bin/env python3
"""Read-only H30 pair + existing attitude math; NEVER constructs a motor bus.

Record decoded sensor quaternion/gyro/acceleration and host timestamps. The H30
driver opens its ports O_RDONLY. Mathematical targets use isolated topics only.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from ament_index_python.packages import get_package_prefix
from sensor_msgs.msg import Imu
from std_msgs.msg import String

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'force_torque_control'))
from imu_node import ImuResearchNode


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(HERE/'calibrated_config.json'))
    p.add_argument('--duration',type=float,default=60.)
    p.add_argument('--reference-mode',default='first_valid')
    p.add_argument('--output',required=True)
    a=p.parse_args();config=json.loads(Path(a.config).read_text())
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    ports=dict(base=config.get('base_imu_port','/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B32028000-if00'),
               platform=config.get('platform_imu_port','/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B0B029000-if00'))
    for port in ports.values():
        if not Path(port).exists():raise RuntimeError('missing IMU port: '+port)
        if subprocess.run(['fuser','-s',port],capture_output=True).returncode!=1:
            raise RuntimeError('IMU port occupied or ownership unavailable: '+port)
    log=(out/'imu.jsonl').open('x',buffering=1);children=[];streams=[]
    counts=dict(base=0,platform=0);first_both=None;reason='duration_complete'
    def record(kind,data):
        log.write(json.dumps(dict(kind=kind,host_monotonic_ns=time.monotonic_ns(),data=data),allow_nan=False)+'\n')
    rclpy.init()
    math_node=ImuResearchNode([Parameter('enabled',value=True),Parameter('config_json',value=json.dumps(config['imu'])),
        Parameter('period_s',value=config['period_s']),Parameter('imu_max_age_s',value=config['imu_max_age_s']),
        Parameter('startup_timeout_s',value=config['startup_timeout_s']),Parameter('reference_mode',value=a.reference_mode)],
        cli_args=['--ros-args','-r','__node:=imu_drift_math',
            '-r','/imu_drift_math/base_imu:=/imu_drift_check/base',
            '-r','/imu_drift_math/platform_imu:=/imu_drift_check/platform',
            '-r','/imu_drift_math/diagnostics:=/imu_drift_check/attitude',
            '-r','/research_force_torque_mock/desired_lengths_m:=/imu_drift_check/lengths'])
    observer=rclpy.create_node('imu_drift_recorder')
    executor=SingleThreadedExecutor();executor.add_node(math_node);executor.add_node(observer)
    subscriptions=[]
    def receive(role,msg):
        counts[role]+=1
        record('raw_'+role,dict(header_ns=int(msg.header.stamp.sec)*10**9+int(msg.header.stamp.nanosec),
            frame_id=msg.header.frame_id,q_xyzw=[msg.orientation.x,msg.orientation.y,msg.orientation.z,msg.orientation.w],
            gyro_rad_s=[msg.angular_velocity.x,msg.angular_velocity.y,msg.angular_velocity.z],
            accel_m_s2=[msg.linear_acceleration.x,msg.linear_acceleration.y,msg.linear_acceleration.z],
            orientation_valid=bool(msg.orientation_covariance[0]!=-1),
            gyro_valid=bool(msg.angular_velocity_covariance[0]!=-1),accel_valid=bool(msg.linear_acceleration_covariance[0]!=-1)))
    def diagnostics(msg):
        try:data=json.loads(msg.data)
        except ValueError:data=msg.data
        record('attitude',data)
    for role in ports:
        subscriptions.append(observer.create_subscription(Imu,'/imu_drift_check/'+role,
            lambda msg,role=role:receive(role,msg),qos_profile_sensor_data))
    subscriptions.append(observer.create_subscription(String,'/imu_drift_check/attitude',diagnostics,10))
    start=time.monotonic()
    try:
        binary=Path(get_package_prefix('h30_imu'))/'lib/h30_imu/h30_imu_node'
        for role,port in ports.items():
            stream=(out/f'h30_{role}.log').open('x');streams.append(stream)
            children.append(subprocess.Popen([str(binary),'--ros-args','-r',f'__node:=imu_drift_{role}',
                '-p',f'port:={port}','-p',f'imu_topic:=/imu_drift_check/{role}',
                '-p',f'frame_id:=force_position_{role}_imu','-p','publish_euler:=false'],stdout=stream,stderr=subprocess.STDOUT))
        while rclpy.ok():
            executor.spin_once(timeout_sec=.01)
            if all(counts.values()) and first_both is None:first_both=time.monotonic()
            if first_both is not None and time.monotonic()-first_both>=a.duration:break
            if first_both is None and time.monotonic()-start>10:raise RuntimeError('missing IMU data')
            if any(c.poll() is not None for c in children):raise RuntimeError('IMU driver exited')
    except KeyboardInterrupt:reason='interrupted'
    except Exception as exc:
        reason=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        for child in children:
            if child.poll() is None:child.terminate()
        for child in children:
            try:child.wait(timeout=3.)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        for stream in streams:stream.close()
        metadata=dict(hardware_write_count=0,motor_serial_opened=False,ports=ports,
            reference_mode=a.reference_mode,counts=counts,duration_s=time.monotonic()-start,
            reason=reason,attitude_fault=math_node.fault,
            timestamp_note='Header is H30 host publish time, not a device sample clock',
            stationary_condition='operator confirmed both sensors fixed; validate with gyro/acceleration')
        (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        executor.shutdown();observer.destroy_node();math_node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
        log.close()
    print(json.dumps(metadata,indent=2),flush=True)


if __name__=='__main__':main()
