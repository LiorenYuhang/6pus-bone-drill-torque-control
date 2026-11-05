#!/usr/bin/env python3
"""Independent dual-IMU/force-position ROS runtime with one real UART owner."""
import argparse
from collections import deque
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
sys.path.insert(0,str(HERE.parent/'force_torque_control'))
sys.path.insert(0,str(HERE))

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.signals import SignalHandlerOptions
from ament_index_python.packages import get_package_prefix
from std_msgs.msg import Empty, Float64MultiArray, String
from pus6_msgs.msg import TimedJointFeedback
from pus6_research_msgs.msg import RawForceFeedback, ForceFeedback
from pus6_motor_driver.uart_bus import UartBus
from imu_node import ImuResearchNode
from controller import Config
from calibration import validate_relative_start_reference
from ros_node import ForcePositionNode
from uart_driver import SixAxisPositionDrive

BASE='/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B32028000-if00'
PLATFORM='/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B0B029000-if00'


def check_free(path):
    if not os.path.exists(path): raise RuntimeError('missing device: '+path)
    result=subprocess.run(['fuser','-s',path],capture_output=True)
    if result.returncode==0: raise RuntimeError('device already in use: '+path)
    if result.returncode!=1: raise RuntimeError('unable to determine device ownership: '+path)


class SessionLog:
    def __init__(self,path):
        self.stream=path.open('x',encoding='utf-8',buffering=1)
        self.lock=threading.Lock()
    def record(self,kind,data):
        row=json.dumps({'kind':kind,'host_monotonic_ns':time.monotonic_ns(),'data':data},allow_nan=False)
        with self.lock: self.stream.write(row+'\n')
    def close(self): self.stream.close()


class UartNode(Node):
    """Subscriber callbacks replace a target slot; worker performs all serial I/O."""
    def __init__(self,config,log,*,active,hold_current,select_servo):
        super().__init__('force_position_uart')
        self.config=config; self.log=log; self.active=active; self.hold=hold_current
        self.select_servo=select_servo
        self.stop_event=threading.Event(); self.target=None; self.target_ns=0
        self.error=None; self.thread=None; self.drive=None; self.bus=None; self.initial=None
        self.stop_reason=None
        self.paused_errors=[]; self.latest_feedback=None
        self.pos=self.create_publisher(TimedJointFeedback,'/force_position/position_feedback',10)
        self.raw=self.create_publisher(RawForceFeedback,'/force_position/raw_force_feedback',10)
        self.force=self.create_publisher(ForceFeedback,'/force_position/force_feedback',10)
        self.relative_force=self.create_publisher(Float64MultiArray,'/force_position/relative_force',10)
        self.diag=self.create_publisher(String,'/force_position/uart_state',10)
        self.create_subscription(Float64MultiArray,'/research_force_position/desired_lengths_m',self.on_target,10)
        self.create_subscription(Empty,'/force_position/stop',self.on_stop,10)
        self.calibration=config.get('calibration')
        self.relative_tared=isinstance(self.calibration,dict) and self.calibration.get('kind')=='relative_tare' and all(
            isinstance(self.calibration.get(n),list) and len(self.calibration[n])==6 and
            all(isinstance(x,(int,float)) and math.isfinite(x) and (x!=0 if n=='relative_scale' else True) for x in self.calibration[n])
            for n in ('raw_zero','relative_scale'))
        self.calibrated=not self.relative_tared and isinstance(self.calibration,dict) and all(
            isinstance(self.calibration.get(n),list) and len(self.calibration[n])==6 and
            all(isinstance(x,(int,float)) and math.isfinite(x) and (x!=0 if n=='newtons_per_raw' else True) for x in self.calibration[n])
            for n in ('raw_zero','newtons_per_raw'))

    def on_target(self,msg):
        # Single immutable pair prevents mixing a timestamp and a different target.
        self.target=(tuple(float(x) for x in msg.data),time.monotonic_ns())

    def on_stop(self,msg):
        self.stop_reason='stop topic requested'
        self.stop_event.set()
        self.log.record('stop_requested',{'source':'/force_position/stop'})

    def publish_snapshot(self,snapshot):
        rows=snapshot['status']; ts=snapshot['rx_ns']; now=time.monotonic_ns()
        p=TimedJointFeedback(); p.header.stamp=self.get_clock().now().to_msg()
        p.snapshot_start_monotonic_ns=min(ts); p.snapshot_end_monotonic_ns=max(ts)
        p.per_axis_rx_done_monotonic_ns=ts; p.raw_steps=[s.current_position for s in rows]
        p.positions_meters=self.drive.positions_m(snapshot)
        p.currents_ma=[s.current_mA for s in rows]; p.forces_g_raw=[s.force_g for s in rows]
        p.fault_codes=[s.error_code for s in rows]; p.temperatures_c=[s.temperature for s in rows]
        f=RawForceFeedback(); f.header.stamp=p.header.stamp; f.source=f.SOURCE_HARDWARE
        f.rx_monotonic_ns=ts; f.force_g_raw=[s.force_g for s in rows]; f.force_adc=[s.force_adc for s in rows]
        f.fault_codes=p.fault_codes
        f.valid=[s.error_code==0 and 0<=s.force_adc<=4095 and 0<=now-t<=self.config['feedback_max_age_s']*1e9 for s,t in zip(rows,ts)]
        values=None; relative=None; delta_raw=None
        if isinstance(self.calibration,dict) and isinstance(self.calibration.get('raw_zero'),list):
            delta_raw=[s.force_g-self.calibration['raw_zero'][i] for i,s in enumerate(rows)]
        if self.relative_tared:
            relative=[delta_raw[i]*self.calibration['relative_scale'][i] for i in range(6)]
            self.relative_force.publish(Float64MultiArray(data=relative))
        if self.calibrated:
            c=self.calibration
            values=[(s.force_g-c['raw_zero'][i])*c['newtons_per_raw'][i] for i,s in enumerate(rows)]
            si=ForceFeedback(); si.header.stamp=p.header.stamp; si.source=si.SOURCE_HARDWARE
            si.force_n=values; si.valid=f.valid; si.rx_monotonic_ns=ts; si.fault_codes=p.fault_codes
            self.force.publish(si)
        self.pos.publish(p); self.raw.publish(f)
        data={'positions_steps':[int(s.current_position) for s in rows],'position_m':list(p.positions_meters),
              'force_raw':[int(s.force_g) for s in rows],'force_adc':[int(s.force_adc) for s in rows],'force_n':values,
              'delta_force_raw':delta_raw,'force_relative':relative,
              'force_unit':'relative_raw' if self.relative_tared else 'N' if self.calibrated else 'unscaled_raw',
              'rx_ns':ts,'fault_codes':[int(s.error_code) for s in rows],'valid':[bool(v) for v in f.valid],
              'active':self.active,'hold_current':self.hold,
              'position_acknowledgements':self.drive.position_acknowledgements}
        self.latest_feedback=data
        self.log.record('feedback',data)
        self.diag.publish(String(data=json.dumps(data,allow_nan=False)))

    def start(self):
        self.thread=threading.Thread(target=self.work,name='force-position-uart',daemon=True)
        self.thread.start()

    def work(self):
        try:
            import serial
            check_free(self.config['port'])
            ser=serial.Serial(self.config['port'],baudrate=self.config['baudrate'],timeout=.03,
                              write_timeout=.03,exclusive=True)
            try: self.bus=UartBus(serial_instance=ser,timeout=.03,retries=0,read_only=not self.active)
            except Exception: ser.close(); raise
            # Per-write history is already persisted by SessionLog; bound only
            # this instance's duplicate in-memory audit during long experiments.
            self.bus.write_audit=deque(maxlen=12)
            self.drive=SixAxisPositionDrive(self.bus,max_start_delta_steps=self.config['max_start_delta_steps'],
                max_cycle_delta_steps=self.config['max_cycle_delta_steps'],event=self.log.record,
                max_speed_m_s=self.config['shaper']['max_speed_m_per_sec'])
            def validate_reference(snapshot):
                if self.active and not self.hold:
                    checked=validate_relative_start_reference(self.config,[s.force_g for s in snapshot['status']])
                    if checked is not None:self.log.record('relative_reference_checked',checked)
            snapshot=self.drive.initialize(active=self.active,select_servo=self.select_servo,
                                           initial_feedback_validator=validate_reference)
            self.initial=self.drive.positions_m(snapshot)
            started=time.monotonic_ns()
            period=self.config['period_s']
            while not self.stop_event.is_set():
                begin=time.monotonic(); target=self.target
                if self.active and self.hold:
                    snapshot=self.drive.command(self.initial)
                elif self.active and target is not None:
                    values,stamp=target
                    if (time.monotonic_ns()-stamp)*1e-9>self.config['target_max_age_s']:
                        raise RuntimeError('position command stream stale')
                    snapshot=self.drive.command(values)
                else:
                    snapshot=self.drive.read()
                    if self.active and (time.monotonic_ns()-started)*1e-9>self.config['startup_timeout_s']:
                        raise RuntimeError('no force-position command received')
                self.publish_snapshot(snapshot)
                self.stop_event.wait(max(0.,period-(time.monotonic()-begin)))
        except Exception as exc:
            self.error=f'{type(exc).__name__}: {exc}'
        finally:
            if self.drive is not None:
                try: self.paused_errors=self.drive.pause_once()
                except Exception as exc: self.paused_errors=[str(exc)]
            if self.paused_errors: self.error=(self.error or '')+'; pause unconfirmed: '+str(self.paused_errors)
            try:
                self.log.record('uart_stopped',{'error':self.error,
                    'register_write_attempts':self.drive.write_attempts if self.drive else 0,
                    'position_acknowledgements':self.drive.position_acknowledgements if self.drive else 0,
                    'pause_confirmed':self.drive.paused and not self.paused_errors if self.drive else False})
            except Exception as exc:
                self.error=(self.error or '')+'; log close: '+str(exc)
            try:
                if self.bus:
                    self.bus.serial.close(); self.bus.close()
            finally:
                self.stop_event.set()

    def close(self):
        self.stop_event.set()
        if self.thread: self.thread.join(timeout=5.)
        if self.thread and self.thread.is_alive(): raise RuntimeError('UART worker did not stop')


def load_config(path,*,active,hold):
    c=json.loads(Path(path).read_text())
    for key in ('period_s','target_max_age_s','feedback_max_age_s','imu_max_age_s','startup_timeout_s'):
        if not isinstance(c.get(key),(int,float)) or not math.isfinite(c[key]) or c[key]<=0:
            raise ValueError(f'{key} must be positive and finite')
    if active and not hold:
        cal=c.get('calibration')
        if not isinstance(cal,dict):
            raise ValueError('calibration required: save a relative reference with calibration.py --tare, or supply known-force calibration')
        if cal.get('kind')=='relative_tare':
            cfg=dict(c['controller'],force_unit='relative_raw',raw_zero=cal.get('raw_zero'),relative_scale=cal.get('relative_scale'))
        else:
            cfg=dict(c['controller'],raw_zero=cal.get('raw_zero'),newtons_per_raw=cal.get('newtons_per_raw'))
        try: Config.from_document(cfg)
        except (KeyError,TypeError,ValueError) as exc: raise ValueError('invalid saved force reference or calibration: '+str(exc)) from exc
        c['control_config']={'controller':cfg,'shaper':c['shaper']}
    return c


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    saved=HERE/'calibrated_config.json'
    p.add_argument('--config',default=str(saved if saved.exists() else HERE/'hardware_config.json'))
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--active',action='store_true',help='send real 0x29 position commands')
    group.add_argument('--read-only',action='store_true',help='real sensors; no register writes')
    p.add_argument('--hold-current',action='store_true',help='active transport check: hold current positions; no force controller')
    p.add_argument('--select-servo',action='store_true',help='select mode 1 at measured positions if currently mode 0')
    p.add_argument('--duration',type=float,default=0.)
    p.add_argument('--log-dir')
    args=p.parse_args(argv)
    if not math.isfinite(args.duration) or args.duration<0: p.error('duration must be nonnegative and finite')
    if (args.hold_current or args.select_servo) and not args.active: p.error('hold-current/select-servo require --active')
    c=load_config(args.config,active=args.active,hold=args.hold_current)
    for path in (c['port'],c.get('base_imu_port',BASE),c.get('platform_imu_port',PLATFORM)): check_free(path)
    run=Path(args.log_dir) if args.log_dir else REPO/'results'/'force_position'/time.strftime('%Y%m%d-%H%M%S')
    run.mkdir(parents=True,exist_ok=False)
    log=SessionLog(run/'telemetry.jsonl')
    (run/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    children=[]; streams=[]; nodes=[]; executor=None; uart=None; imu=None; control=None; code=0; reason='normal'
    # Keep the ROS context alive until the UART worker has attempted its pause.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    def request_signal_stop(signum,frame):
        if uart is None: raise KeyboardInterrupt
        uart.stop_reason=f'signal {signum}'
        uart.stop_event.set()
    signal.signal(signal.SIGINT,request_signal_stop)
    signal.signal(signal.SIGTERM,request_signal_stop)
    try:
        imu=ImuResearchNode([Parameter('enabled',value=True),Parameter('config_json',value=json.dumps(c['imu'])),
            Parameter('period_s',value=c['period_s']),Parameter('imu_max_age_s',value=c['imu_max_age_s']),
            Parameter('startup_timeout_s',value=c['startup_timeout_s']),Parameter('reference_mode',value=c.get('imu_reference_mode','stationary')),
            Parameter('publish_unshaped_ik',value=True)],cli_args=['--ros-args',
                '-r','/research_force_torque_mock/desired_lengths_m:=/force_position/ik_lengths_m',
                '-r','/research_imu_lengths/base_imu:=/force_position/base_imu',
                '-r','/research_imu_lengths/platform_imu:=/force_position/platform_imu',
                '-r','/research_imu_lengths/diagnostics:=/force_position/attitude'])
        nodes.append(imu)
        imu.sample_observer=lambda role,data:log.record('imu_'+role,data)
        uart=UartNode(c,log,active=args.active,hold_current=args.hold_current,select_servo=args.select_servo)
        nodes.append(uart)
        if args.active and not args.hold_current:
            control=ForcePositionNode([Parameter('config_json',value=json.dumps(c['control_config'])),
                Parameter('desired_topic',value='/force_position/ik_lengths_m'),
                Parameter('position_topic',value='/force_position/position_feedback'),
                Parameter('force_topic',value='/force_position/raw_force_feedback'),
                Parameter('anchor_to_first_feedback',value=True),Parameter('target_max_age_s',value=c['target_max_age_s'])])
            nodes.append(control)
        def log_text(kind,msg):
            try: data=json.loads(msg.data)
            except ValueError: data=msg.data
            if kind=='attitude' and isinstance(data,dict) and 'hardware_write_count' in data:
                data['imu_component_hardware_write_count']=data.pop('hardware_write_count')
            log.record(kind,data)
        uart.create_subscription(String,'/force_position/attitude',lambda m:log_text('attitude',m),10)
        uart.create_subscription(String,'/research_force_position/diagnostics',lambda m:log_text('control',m),10)
        binary=Path(get_package_prefix('h30_imu'))/'lib/h30_imu/h30_imu_node'
        for role,path in (('base',c.get('base_imu_port',BASE)),('platform',c.get('platform_imu_port',PLATFORM))):
            stream=(run/f'h30_{role}.log').open('x'); streams.append(stream)
            children.append(subprocess.Popen([str(binary),'--ros-args','-r',f'__node:=force_position_{role}_imu',
                '-p',f'port:={path}','-p',f'imu_topic:=/force_position/{role}_imu',
                '-p',f'frame_id:=force_position_{role}_imu','-p','publish_euler:=false'],stdout=stream,stderr=subprocess.STDOUT))
        executor=SingleThreadedExecutor()
        for node in nodes: executor.add_node(node)
        log.record('start',{'active':args.active,'hold_current':args.hold_current,'pid':os.getpid()})
        print(f'LOG_DIR={run}\nMODE={"active-current-hold" if args.hold_current else "active-force-position" if args.active else "real-sensors-read-only"}',flush=True)
        uart.start(); started=time.monotonic()
        while rclpy.ok() and not uart.stop_event.is_set():
            executor.spin_once(timeout_sec=.01)
            if imu.fault or control and control.fault: raise RuntimeError(imu.fault or control.fault)
            if any(child.poll() is not None for child in children): raise RuntimeError('IMU process exited')
            if args.duration and time.monotonic()-started>=args.duration: break
        if uart.error: raise RuntimeError(uart.error)
        if uart.stop_reason: reason=uart.stop_reason
    except KeyboardInterrupt: reason='interrupted'
    except Exception as exc: reason=str(exc); code=2
    finally:
        if uart:
            try: uart.close()
            except Exception as exc: reason=str(exc); code=2
            if uart.error: reason=uart.error; code=2
        for child in children:
            if child.poll() is None: child.terminate()
        for child in children:
            try: child.wait(timeout=3.)
            except subprocess.TimeoutExpired: child.kill(); child.wait()
        for stream in streams: stream.close()
        if executor: executor.shutdown()
        for node in nodes: node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
        log.record('stop',{'reason':reason,'exit_code':code}); log.close()
    print(f'STOP {reason}',flush=True)
    return code


if __name__=='__main__': sys.exit(main())
