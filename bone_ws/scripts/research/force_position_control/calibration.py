#!/usr/bin/env python3
"""Save a stationary relative zero, or fit an affine calibration from known loads."""
import argparse
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

GF_TO_N=0.00980665
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]


def validate_relative_start_reference(config,actual_raw):
    """Reject a mismatched relative reference before any Servo/motion writes.

    Use the existing experiment's force range, not a date-based expiry or an
    invented calibration. Never automatically erase load or change PID gains.
    """
    cal=config.get('calibration',{})
    if cal.get('kind')!='relative_tare':return None
    delta=[(v-z)*s for v,z,s in zip(actual_raw,cal['raw_zero'],cal['relative_scale'])]
    limits=config['controller']['force_limit_relative']
    if len(delta)!=6 or any(not math.isfinite(v) or abs(v)>limit for v,limit in zip(delta,limits)):
        raise ValueError('saved relative force reference does not match current initial load: '
            f'offsets={delta}, existing force range={limits}; explicitly refresh the software relative reference '
            'with calibration.py --tare (reads only), or verify the intended loaded reference; no automatic re-zero')
    return dict(initial_offset_relative=delta,force_limit_relative=limits)


def relative_config(config,zero,measurement):
    """Define a relative reference; no claim of zero absolute axial load or N."""
    c=dict(config); ctrl=dict(c['controller'])
    old=c.get('calibration') or {}
    scales=old.get('relative_scale',[1.]*6)
    if ctrl.get('force_unit','N')!='relative_raw':
        # Change the numerical force unit for initial gains using the manual's
        # nominal gf conversion. This is gain selection, not sensor calibration.
        for name in ('kp','ki','kd'): ctrl[name]=[v/GF_TO_N for v in ctrl[name]]
        ctrl['force_limit_relative']=[v/GF_TO_N for v in ctrl.pop('force_limit_n')]
        ctrl['admittance_m_per_relative_s']=[v*GF_TO_N for v in ctrl.pop('admittance_m_per_ns')]
    ctrl['force_unit']='relative_raw'
    c['controller']=ctrl
    c['calibration']={'kind':'relative_tare','raw_zero':zero,'relative_scale':scales,
                      'absolute_force_calibrated':False,'measurement':measurement}
    return c


def collect_tare(config,duration):
    from pus6_motor_driver.uart_bus import UartBus
    from uart_driver import SixAxisPositionDrive
    import serial
    port=config['port']
    owner=subprocess.run(['fuser','-s',port],capture_output=True)
    if owner.returncode!=1: raise RuntimeError('UART already in use or ownership check failed: '+port)
    run=REPO/'results'/'force_position'/('zero_'+time.strftime('%Y%m%d-%H%M%S'))
    run.mkdir(parents=True,exist_ok=False)
    columns=[[] for _ in range(6)]; positions=[[] for _ in range(6)]
    ser=serial.Serial(port,baudrate=config['baudrate'],timeout=.03,write_timeout=.03,exclusive=True)
    try: bus=UartBus(serial_instance=ser,timeout=.03,retries=0,read_only=True)
    except Exception: ser.close(); raise
    try:
        drive=SixAxisPositionDrive(bus,max_start_delta_steps=config['max_start_delta_steps'],
                    max_cycle_delta_steps=config['max_cycle_delta_steps'],event=lambda *_:None)
        begin=time.monotonic()
        with (run/'telemetry.jsonl').open('x') as stream:
            while time.monotonic()-begin<duration:
                cycle_start=time.monotonic()
                sample=drive.read(); raw=[s.force_g for s in sample['status']]
                steps=[s.current_position for s in sample['status']]
                for i in range(6): columns[i].append(raw[i]); positions[i].append(steps[i])
                row={'kind':'feedback','host_monotonic_ns':time.monotonic_ns(),'data':{
                    'force_raw':raw,'force_adc':[s.force_adc for s in sample['status']],
                    'positions_steps':steps,'rx_ns':sample['rx_ns'],'valid':[True]*6,'fault_codes':[0]*6}}
                stream.write(json.dumps(row)+'\n')
                time.sleep(max(0.,config['period_s']-(time.monotonic()-cycle_start)))
        zero=[statistics.mean(v) for v in columns]
        measurement={'reference':'user-defined static horizontal loaded reference, not absolute zero force',
            'captured_at':time.strftime('%Y-%m-%d %H:%M:%S %z'),'samples_per_axis':[len(v) for v in columns],
            'duration_s':time.monotonic()-begin,'raw_std':[statistics.pstdev(v) for v in columns],
            'raw_min':[min(v) for v in columns],'raw_max':[max(v) for v in columns],
            'position_min_steps':[min(v) for v in positions],'position_max_steps':[max(v) for v in positions],
            'register_write_count':len(bus.write_audit),'data_dir':str(run)}
        (run/'zero_result.json').write_text(json.dumps({'raw_zero':zero,**measurement},indent=2)+'\n')
        return zero,measurement
    finally:
        ser.close(); bus.close()


def raw_means(run):
    path=Path(run)
    if path.is_dir(): path=path/'telemetry.jsonl'
    columns=[[] for _ in range(6)]
    with path.open() as stream:
        for line in stream:
            row=json.loads(line)
            if row.get('kind')!='feedback': continue
            data=row['data']
            if len(data['force_raw'])!=6 or not all(data['valid']) or any(data['fault_codes']):
                continue
            for values,value in zip(columns,data['force_raw']): values.append(float(value))
    if any(not values for values in columns): raise ValueError('no valid six-axis force samples')
    return [statistics.mean(v) for v in columns],[statistics.pstdev(v) for v in columns],len(columns[0])


def affine_calibration(first_raw,second_raw,first_force_n,second_force_n):
    arrays=(first_raw,second_raw,first_force_n,second_force_n)
    if any(len(v)!=6 or not all(math.isfinite(float(x)) for x in v) for v in arrays):
        raise ValueError('six finite values required for each calibration state')
    zero=[]; scale=[]
    for i,(r0,r1,f0,f1) in enumerate(zip(*arrays)):
        if r1==r0 or f1==f0:
            raise ValueError(f'axis {i+1}: two distinct known force and raw values required')
        slope=(f1-f0)/(r1-r0)
        intercept_zero=r0-f0/slope
        if not math.isfinite(slope) or not math.isfinite(intercept_zero):
            raise ValueError('nonfinite calibration')
        scale.append(slope); zero.append(intercept_zero)
    return {'raw_zero':zero,'newtons_per_raw':scale}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(HERE/'hardware_config.json'))
    p.add_argument('--tare',action='store_true',help='capture current loaded state as a relative zero; UART reads only')
    p.add_argument('--duration',type=float,default=8.)
    p.add_argument('--replace',action='store_true',help='replace the explicitly selected output config')
    p.add_argument('--first-run')
    p.add_argument('--second-run')
    p.add_argument('--first-force-n',type=float,nargs=6,metavar='N')
    p.add_argument('--second-force-n',type=float,nargs=6,metavar='N')
    p.add_argument('--output',default=str(HERE/'calibrated_config.json'))
    a=p.parse_args()
    if Path(a.output).exists() and not a.replace: p.error('output exists; use --replace to update saved zero explicitly')
    if not math.isfinite(a.duration) or a.duration<=0: p.error('positive finite duration required')
    if a.tare:
        c=json.loads(Path(a.config).read_text())
        zero,measurement=collect_tare(c,a.duration)
        c=relative_config(c,zero,measurement)
        with Path(a.output).open('w' if a.replace else 'x') as stream:
            json.dump(c,stream,indent=2,allow_nan=False); stream.write('\n')
        print(json.dumps({'output':str(a.output),'raw_zero':zero,**measurement},indent=2))
        return
    if any(v is None for v in (a.first_run,a.second_run,a.first_force_n,a.second_force_n)):
        p.error('known-force fitting requires both runs and both six-axis force arrays; use --tare for relative zero')
    first,noise0,n0=raw_means(a.first_run); second,noise1,n1=raw_means(a.second_run)
    c=json.loads(Path(a.config).read_text())
    c['calibration']=affine_calibration(first,second,a.first_force_n,a.second_force_n)
    c['calibration']['measurement']={'first_run':str(a.first_run),'second_run':str(a.second_run),
        'first_known_force_n':a.first_force_n,'second_known_force_n':a.second_force_n,
        'raw_mean_first':first,'raw_mean_second':second,'raw_std_first':noise0,'raw_std_second':noise1,
        'sample_counts':[n0,n1],
        'positive_force':'user supplied signed axial forces in the chosen extension-force convention'}
    with Path(a.output).open('w' if a.replace else 'x') as stream: json.dump(c,stream,indent=2,allow_nan=False); stream.write('\n')
    print(json.dumps(c['calibration'],indent=2))


if __name__=='__main__': main()
