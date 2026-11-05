"""Six-axis position transport over one caller-owned UartBus, no J14 dependency."""
import math
import time
from pus6_motor_driver.motor import Motor
from pus6_motor_driver.safety import validate_status_feedback
from pus6_kinematics.geometry_config import load_geometry_config
from controller import position_bounds_from_geometry


class SixAxisPositionDrive:
    def __init__(self, bus, *, max_start_delta_steps, max_cycle_delta_steps, event, clock=time.monotonic_ns,
                 max_speed_m_s=None):
        for value in (v for v in (max_cycle_delta_steps,max_start_delta_steps) if v is not None):
            if not isinstance(value,int) or value<=0:
                raise ValueError('positive integer motion limits required')
        self.bus=bus; self.event=event; self.clock=clock
        self.motors=[Motor(bus,i) for i in range(1,7)]
        self.geometry=load_geometry_config()
        self.max_start=max_start_delta_steps; self.max_cycle=max_cycle_delta_steps
        if max_speed_m_s is not None:
            if isinstance(max_speed_m_s,(int,float)): max_speed_m_s=[max_speed_m_s]*6
            if len(max_speed_m_s)!=6 or any(not math.isfinite(v) or v<=0 for v in max_speed_m_s):
                raise ValueError('six positive finite speed limits required')
            max_speed_m_s=tuple(max_speed_m_s)
        self.max_speed=max_speed_m_s; self.dispatched_m=None; self.last_command_ns=None
        self.anchor=None; self.previous=None; self.lower=[]; self.upper=[]; self.modes=[]
        self.wrote=False; self.paused=False; self.fault=None
        self.write_attempts=0; self.position_acknowledgements=0

    def _write(self,motor,register,values):
        self.wrote=True; self.write_attempts+=1
        # Record before the single attempt; Motor.write_register explicitly uses retries=0.
        self.event('uart_write',{'axis':motor.motor_id,'register':register,'values':list(values),'attempt':self.write_attempts})
        result=motor.write_register(register,values)
        validate_status_feedback(result,motor.motor_id)
        return result

    def initialize(self, *, active=False, select_servo=False, initial_feedback_validator=None):
        snapshot=self.read()
        self.anchor=[s.current_position for s in snapshot['status']]
        self.previous=self.anchor[:]
        for m in self.motors:
            self.modes.append(int(m.read_register(0x25).values[0]))
            self.upper.append(int(m.read_register(0x23).values[0]))
            self.lower.append(int(m.read_register(0x24).values[0]))
        if any(not lo<=p<=hi for lo,p,hi in zip(self.lower,self.anchor,self.upper)):
            raise ValueError('initial position outside drive limits')
        self.event('drive_initial',{'positions_steps':self.anchor,'modes':self.modes,
                                  'lower_steps':self.lower,'upper_steps':self.upper})
        if initial_feedback_validator is not None:initial_feedback_validator(snapshot)
        if active:
            if any(mode not in (0,1) for mode in self.modes):
                raise ValueError('active drive is in a non-position mode')
            if any(mode!=1 for mode in self.modes):
                if not select_servo:
                    raise ValueError('servo mode 0x25=1 required; use --select-servo to select it at current positions')
                # In mode 0 first set the latest measured position, then select mode 1.
                # Never write force, voltage, speed, save, homing or reset registers.
                for i,m in enumerate(self.motors):
                    if self.modes[i]!=1:
                        current=m.read_status()
                        if current.error_code or abs(current.current_position-self.anchor[i])>1:
                            raise ValueError(f'axis {i+1}: initial position changed')
                        self._write(m,0x29,[current.current_position])
                        self._write(m,0x25,[1])
                        # As in the existing J7 configure_servo path, rely on the
                        # acknowledged mode write: firmware register readback is
                        # not used by the established position-servo entry.
                        self.modes[i]=1
                        self.event('servo_mode_write_acknowledged',{'axis':i+1,'mode_commanded':1})
        self.dispatched_m=self.positions_m(snapshot)
        self.last_command_ns=self.clock()
        return snapshot

    def read(self):
        rows=[]; times=[]
        for m in self.motors:
            s=m.read_status(); times.append(self.clock()); rows.append(s)
            validate_status_feedback(s,m.motor_id)
            if not 0<=s.force_adc<=4095: raise RuntimeError(f'axis {m.motor_id}: invalid force ADC')
        return {'status':rows,'rx_ns':times}

    def positions_m(self,snapshot):
        g=self.geometry
        return [g.operating_home_stroke_m[i]+(s.current_position-g.operating_home_raw_steps[i])*g.meters_per_step
                for i,s in enumerate(snapshot['status'])]

    def position_bounds_m(self):
        return position_bounds_from_geometry(self.geometry,self.lower,self.upper)

    def command(self, positions_m):
        if self.fault or self.paused or self.anchor is None or self.modes!=[1]*6:
            raise RuntimeError('position drive not active')
        if len(positions_m)!=6 or not all(math.isfinite(float(x)) for x in positions_m):
            raise ValueError('six finite position targets required')
        if self.max_cycle is None and self.max_speed is None:
            raise ValueError('a time-based speed or per-cycle limit is required')
        g=self.geometry
        now=self.clock()
        lower,upper=self.position_bounds_m()
        # Reject an out-of-stroke request before any speed shaping or UART write.
        requested_steps=[int(round(g.operating_home_raw_steps[i]+(float(x)-g.operating_home_stroke_m[i])/g.meters_per_step))
               for i,x in enumerate(positions_m)]
        for i,s in enumerate(requested_steps):
            encoded_m=g.operating_home_stroke_m[i]+(s-g.operating_home_raw_steps[i])*g.meters_per_step
            if (not math.nextafter(lower[i],-math.inf)<=encoded_m<=math.nextafter(upper[i],math.inf)
                    or not self.lower[i]<=s<=self.upper[i]):
                raise ValueError(f'axis {i+1}: position target exceeds drive limits')
        dispatched=[float(x) for x in positions_m]
        if self.max_speed is not None:
            dt=(now-self.last_command_ns)*1e-9
            if dt<=0: raise ValueError('nonpositive command interval')
            dispatched=[old+max(-v*dt,min(v*dt,target-old))
                        for old,target,v in zip(self.dispatched_m,dispatched,self.max_speed)]
        steps=[int(round(g.operating_home_raw_steps[i]+(x-g.operating_home_stroke_m[i])/g.meters_per_step))
               for i,x in enumerate(dispatched)]
        # Validate all six before the first write. No partial validation/writes.
        for i,s in enumerate(steps):
            # Check the target actually encoded on the wire. Sub-step shaper
            # residuals at an endpoint must not reject a valid 0/2000-step target.
            encoded_m=g.operating_home_stroke_m[i]+(s-g.operating_home_raw_steps[i])*g.meters_per_step
            if (not math.nextafter(lower[i],-math.inf)<=encoded_m<=math.nextafter(upper[i],math.inf)
                    or not self.lower[i]<=s<=self.upper[i]
                    or (self.max_start is not None and abs(s-self.anchor[i])>self.max_start)
                    or (self.max_cycle is not None and abs(s-self.previous[i])>self.max_cycle)):
                raise ValueError(f'axis {i+1}: position target exceeds displacement or drive limits')
        rows=[]; times=[]
        try:
            for m,s in zip(self.motors,steps):
                rows.append(self._write(m,0x29,[s])); times.append(self.clock())
                self.position_acknowledgements+=1
            self.previous=steps
            self.dispatched_m=dispatched; self.last_command_ns=now
            self.event('position_acknowledged',{'targets_steps':steps,'requested_steps':requested_steps,'rx_ns':times})
            return {'status':rows,'rx_ns':times}
        except Exception as exc:
            self.fault=str(exc)
            self.pause_once()
            raise

    def pause_once(self):
        if self.paused or not self.wrote: return []
        self.paused=True; errors=[]
        for m in self.motors:
            try:
                # A failed log write cannot prevent this single pause attempt.
                self.write_attempts+=1
                s=m.write_register(0x1a,[1])
                if s.error_code: raise RuntimeError(f'drive fault {s.error_code}')
            except Exception as exc: errors.append(f'axis {m.motor_id}: {exc}')
        self.event('pause_complete',{'errors':errors,'confirmed':not errors,'write_attempts':self.write_attempts})
        return errors
