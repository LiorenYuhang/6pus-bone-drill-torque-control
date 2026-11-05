#!/usr/bin/env python3
"""ROS IMUs -> research target lengths; all outputs are non-actuating."""
import json
import math
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64MultiArray, String
from imu_chain import ImuToLengths, quat, StationaryImuReference


def imu_message_fields(msg):
    """Decoded hardware data; preserve raw sensor orientation and valid flags."""
    return dict(header_ns=int(msg.header.stamp.sec)*1000000000+int(msg.header.stamp.nanosec),
        frame_id=msg.header.frame_id,
        q_xyzw=[float(v) for v in (msg.orientation.x,msg.orientation.y,msg.orientation.z,msg.orientation.w)],
        gyro_rad_s=[float(v) for v in (msg.angular_velocity.x,msg.angular_velocity.y,msg.angular_velocity.z)],
        accel_m_s2=[float(v) for v in (msg.linear_acceleration.x,msg.linear_acceleration.y,msg.linear_acceleration.z)],
        orientation_valid=bool(msg.orientation_covariance[0]!=-1),gyro_valid=bool(msg.angular_velocity_covariance[0]!=-1),
        accel_valid=bool(msg.linear_acceleration_covariance[0]!=-1))

class ImuResearchNode(Node):
    def __init__(self, parameter_overrides=None, cli_args=None):
        super().__init__('research_imu_lengths',parameter_overrides=parameter_overrides,cli_args=cli_args)
        self.declare_parameter('enabled',False); self.declare_parameter('config_json','{}')
        self.declare_parameter('period_s',.02); self.declare_parameter('imu_max_age_s',.2)
        self.declare_parameter('startup_timeout_s',.2); self.declare_parameter('reference_mode','configured')
        self.declare_parameter('publish_unshaped_ik',False)
        self.reference_captured=False
        self.reference_capture=StationaryImuReference()
        self.measurements={};self.sample_observer=None
        self.base=self.platform=None; self.start_ns=time.monotonic_ns(); self.fault=None
        self.pub=self.create_publisher(Float64MultiArray,'/research_force_torque_mock/desired_lengths_m',10)
        self.diag=self.create_publisher(String,'~/diagnostics',10)
        self.create_subscription(Imu,'~/base_imu',lambda msg:self.receive('base',msg),qos_profile_sensor_data)
        self.create_subscription(Imu,'~/platform_imu',lambda msg:self.receive('platform',msg),qos_profile_sensor_data)
        self.chain=None
        if self.get_parameter('enabled').value:
            self.chain=ImuToLengths(**json.loads(self.get_parameter('config_json').value))
            self.create_timer(float(self.get_parameter('period_s').value),self.tick)
    def receive(self,name,msg):
        try:
            data=imu_message_fields(msg);data['rx_monotonic_ns']=time.monotonic_ns()
            if self.sample_observer is not None:self.sample_observer(name,data)
            if msg.orientation_covariance[0] == -1: raise ValueError('IMU orientation unavailable')
            q=msg.orientation; q=quat((q.x,q.y,q.z,q.w))
            stamp_ns=int(msg.header.stamp.sec)*1000000000+int(msg.header.stamp.nanosec)
            age_ns=self.get_clock().now().nanoseconds-stamp_ns
            max_age=float(self.get_parameter('imu_max_age_s').value)*1e9
            if stamp_ns<=0 or age_ns<0 or age_ns>max_age: raise ValueError('IMU header timestamp invalid or stale')
            prior=getattr(self,name)
            if prior is not None and stamp_ns<=prior[2]:return
            setattr(self,name,(q,data['rx_monotonic_ns'],stamp_ns));self.measurements[name]=data
            if self.get_parameter('reference_mode').value=='stationary' and not self.reference_captured:
                self.reference_capture.add(name,q,data['gyro_rad_s'] if data['gyro_valid'] else None,stamp_ns)
        except ValueError as exc:
            self.fault=str(exc); self.diag.publish(String(data='FAULT: '+self.fault))
    def tick(self):
        if self.fault or self.chain is None: return
        now=time.monotonic_ns(); maxage=float(self.get_parameter('imu_max_age_s').value)
        try:
            if self.base is None or self.platform is None:
                if (now-self.start_ns)*1e-9>float(self.get_parameter('startup_timeout_s').value): raise ValueError('IMU feedback missing')
                return
            ros_now=self.get_clock().now().nanoseconds
            if any((now-s[1])*1e-9>maxage or not 0<=ros_now-s[2]<=maxage*1e9 for s in (self.base,self.platform)):
                raise ValueError('IMU feedback stale')
            if not self.reference_captured:
                mode=self.get_parameter('reference_mode').value
                if mode=='first_valid': self.chain.base0=self.base[0]; self.chain.platform0=self.platform[0]
                elif mode=='stationary':
                    capture=self.reference_capture.capture()
                    if capture is None:
                        if (now-self.start_ns)*1e-9>float(self.get_parameter('startup_timeout_s').value):
                            raise ValueError('stationary IMU reference unavailable: '+self.reference_capture.reason)
                        return
                    self.chain.base0=tuple(capture['base']['q_xyzw']);self.chain.platform0=tuple(capture['platform']['q_xyzw'])
                    self.diag.publish(String(data=json.dumps({'stationary_reference':capture})))
                elif mode!='configured': raise ValueError('invalid reference mode')
                self.reference_captured=True
                self.diag.publish(String(data=json.dumps({'reference_mode':mode,'base_reference_xyzw':self.chain.base0,'platform_reference_xyzw':self.chain.platform0})))
            result=self.chain.step(self.base[0],self.platform[0],now)
            if result is None: return
            result.update(base_rx_ns=self.base[1],platform_rx_ns=self.platform[1],base_header_ns=self.base[2],platform_header_ns=self.platform[2],frame='pus6_base',hardware_write_count=0)
            result.update(pair_header_skew_s=abs(self.base[2]-self.platform[2])*1e-9,
                base_raw=self.measurements.get('base'),platform_raw=self.measurements.get('platform'))
            target=result['ik_lengths_m'] if self.get_parameter('publish_unshaped_ik').value else result['desired_lengths_m']
            self.pub.publish(Float64MultiArray(data=target))
            self.diag.publish(String(data=json.dumps(result)))
        except Exception as exc:
            self.fault=str(exc); self.diag.publish(String(data='FAULT: '+self.fault))

def main():
    rclpy.init(); node=ImuResearchNode()
    try: rclpy.spin(node)
    finally: node.destroy_node(); rclpy.shutdown()
if __name__=='__main__': main()
