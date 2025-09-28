import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Qt5Agg")
from dataclasses import dataclass, field
from collections import deque
from typing import List, Tuple, Callable, Optional
from scipy.interpolate import interp1d
import pandas as pd

@dataclass
class PIDController:
    kp: float  # Proportional gain
    ki: float  # Integral gain
    kd: float  # Derivative gain

    def __post_init__(self):
        self.integral = 0
        self.previous_error = 0

    def compute(self, error, dt):
        """Compute PID output for a given error and time step."""
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt if dt > 0 else 0
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        self.previous_error = error
        return output

@dataclass
class Car:
    car_id: int
    initial_velocity: float
    initial_distance: float
    max_velocity: float
    max_acceleration: float
    min_acceleration: float
    desired_distance: float
    desired_headway: float
    strategy: str
    velocity_profile: Optional[Callable[[float], float]] = None
    logs: List[Tuple[float, float, float, float, float, float]] = field(default_factory=list)
    pid_params: Tuple[float, float, float] = (1.3, 0.3, 0.2)
    pid_controller: PIDController = field(init=False)
    traffic_state: str = 'normal'
    velocity: float = 0
    distance_travelled: float = 0
    simulation_time: float = 0
    velocity_calculation_strategy: Callable = None
    leader_distances: deque = field(default_factory=lambda: deque(maxlen=30))
    leader_velocities: deque = field(default_factory=lambda: deque(maxlen=30))
    my_distances: deque = field(default_factory=lambda: deque(maxlen=30))
    my_velocities: deque = field(default_factory=lambda: deque(maxlen=30))
    dt: float = 0.1  # Simulation time step
    reaction_time: float = 1.0  # Driver's reaction time in seconds
    reaction_index: int = field(init=False)

    def __post_init__(self):
        
        self.pid_controller = PIDController(*self.pid_params)
        self.distance_travelled = self.initial_distance
        self.reaction_index = int(self.reaction_time / self.dt)
        
        if self.car_id == 0:
            self.create_velocity_profile()

        if self.strategy == 'pid-v':
            self.velocity_calculation_strategy = self.pid_velocity
        elif self.strategy == 'ghr':
            self.velocity_calculation_strategy = self.ghr_model
        elif self.strategy == 'gipps':
            self.velocity_calculation_strategy = self.gipps_model
        elif self.strategy == 'idm':
            self.velocity_calculation_strategy = self.idm_model
        elif self.strategy == 'acc':
            self.velocity_calculation_strategy = self.acc_model

    def create_velocity_profile(self):
        
        if self.traffic_state == 'free':
            time_points = [0.0, 3.0, 4.0, 48.0, 49.0, 60.0]
            velocities = [0.0, 0.0, 0.7, 0.7, 0.0, 0.0]
        elif self.traffic_state == 'normal':
            time_points = [0.0, 5.0, 6.0, 16.0, 17.0, 22.0, 22.5, 32.5, 33.0, 38.0, 38.2, 48.2, 48.3, 60.0]
            velocities = [0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0]
        elif self.traffic_state == 'congestion':
            time_points = [0.0, 2.0, 3.0, 8.0, 9.0, 14.0, 15.0, 19.0, 20.0, 25.0, 26.0, 29.0, 30.0, 35.0, 36.0, 40.0, 41.0,
                           46.0, 47.0, 50.0, 51.0, 60.0]
            velocities = [0.0, 0.0, 0.25, 0.25, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0,
                          0.25, 0.25, 0.0, 0.0]
        self.velocity_profile = interp1d(time_points, velocities, kind='linear', fill_value="extrapolate")

    def pid_velocity(self, leader_distance: float, leader_velocity: float, dt: float):
        distance_error = (leader_distance - self.distance_travelled) - self.desired_distance
        control_velocity = self.pid_controller.compute(distance_error, dt)
        max_vel_a = self.velocity + self.max_acceleration * dt
        min_vel_a = self.velocity + self.min_acceleration * dt
        max_vel = np.minimum(max_vel_a, self.max_velocity)
        min_vel = np.maximum(min_vel_a, 0.0)
        return np.clip(control_velocity, min_vel, max_vel)

    def ghr_model(self, leader_distance: float, leader_velocity: float, dt: float):
    
        # Constants
        sensitivity_c = 4.0
        sensitivity_m = 1e-8
        sensitivity_l = 0.5
        
        if len(self.my_distances) > self.reaction_index and len(self.my_velocities) > self.reaction_index:
            reaction_my_distance = self.my_distances[-self.reaction_index]
            reaction_my_velocity = self.my_velocities[-self.reaction_index]
            reaction_leader_distance = self.leader_distances[-self.reaction_index]
            reaction_leader_velocity = self.leader_velocities[-self.reaction_index]
        else:
            reaction_my_distance = self.distance_travelled
            reaction_my_velocity = self.velocity
            reaction_leader_distance = leader_distance
            reaction_leader_velocity = leader_velocity
    
        relative_speed = reaction_leader_velocity - reaction_my_velocity
        relative_distance = reaction_leader_distance - reaction_my_distance
        acceleration = sensitivity_c * self.velocity ** sensitivity_m * relative_speed / (relative_distance ** sensitivity_l)
        acceleration = np.clip(acceleration, self.min_acceleration, self.max_acceleration)
        new_velocity = np.clip(self.velocity + acceleration * dt, 0.0, self.max_velocity)
    
        return new_velocity
    
    def gipps_model(self, leader_distance: float, leader_velocity: float, dt: float):
        
        if len(self.my_distances) > self.reaction_index and len(self.my_velocities) > self.reaction_index:
            reaction_my_distance = self.my_distances[-self.reaction_index]
            reaction_my_velocity = self.my_velocities[-self.reaction_index]
            reaction_leader_distance = self.leader_distances[-self.reaction_index]
            reaction_leader_velocity = self.leader_velocities[-self.reaction_index]
        else:
            reaction_my_distance = self.distance_travelled
            reaction_my_velocity = self.velocity
            reaction_leader_distance = leader_distance
            reaction_leader_velocity = leader_velocity
    
        relative_distance = reaction_leader_distance - reaction_my_distance
        b = np.abs(self.min_acceleration)
        sqrt0 = b**2 * self.reaction_time**2 + reaction_leader_velocity**2 + 2.0 * b * (relative_distance - self.desired_distance)
        if sqrt0 < 0.0:
            sqrt0 = 0.0
        # Safety distance
        vs = -b * self.reaction_time + np.sqrt(sqrt0)
        max_vel_a = self.velocity + self.max_acceleration * dt
        min_vel_a = self.velocity + self.min_acceleration * dt
        max_vel = np.minimum(max_vel_a, self.max_velocity)
        min_vel = np.maximum(min_vel_a, 0.0)
        new_velocity = np.clip(vs, min_vel, max_vel)
        
        return new_velocity
    
    def idm_model(self, leader_distance: float, leader_velocity: float, dt: float):
        # Constants
        delta = 2.0  # Acceleration exponent
        relative_distance = leader_distance - self.distance_travelled
        relative_speed = self.velocity - leader_velocity
        b = np.abs(self.min_acceleration)
        de = self.desired_distance + max(0.0, self.velocity * self.desired_headway +
                                         (self.velocity * relative_speed) / (2.0 * np.sqrt(self.max_acceleration * b)))
        acceleration = self.max_acceleration * (1 - (self.velocity / self.max_velocity) ** delta - (de / relative_distance) ** 2)
        acceleration = np.clip(acceleration, self.min_acceleration, self.max_acceleration)
        new_velocity = np.clip(self.velocity + acceleration * dt, 0.0, self.max_velocity)
        
        return new_velocity
    
    def acc_model(self, leader_distance: float, leader_velocity: float, dt: float):
        # Constants
        k1 = 3.0
        k2 = 4.0
    
        relative_distance = leader_distance - self.distance_travelled
        relative_speed = leader_velocity - self.velocity
        acceleration = k1 * (relative_distance - self.desired_distance - self.desired_headway * self.velocity) + k2 * relative_speed
        acceleration = np.clip(acceleration, self.min_acceleration, self.max_acceleration)
        new_velocity = np.clip(self.velocity + acceleration * dt, 0.0, self.max_velocity)
        
        return new_velocity

    def update(self, dt, leader_distance=None, leader_velocity=None):
        self.my_distances.append(self.distance_travelled)
        self.my_velocities.append(self.velocity)

        self.leader_distances.append(leader_distance)
        self.leader_velocities.append(leader_velocity)
        
        self.logs.append((self.simulation_time, self.distance_travelled, self.velocity, 0, leader_distance, leader_velocity))

        if self.velocity_profile and self.car_id == 0:
            self.velocity = self.velocity_profile(self.simulation_time)
        elif leader_distance is not None and self.velocity_calculation_strategy:
            self.velocity = self.velocity_calculation_strategy(leader_distance, leader_velocity, dt)

        self.distance_travelled += self.velocity * dt
        self.simulation_time += dt

@dataclass
class Simulation:
    cars: List[Car]
    simulation_time: float
    dt: float = 0.1
    current_time: float = field(default=0, init=False)

    def run(self):
        """Run the simulation, updating each car's state for the duration of the simulation."""
        while self.current_time < self.simulation_time:
            for i, car in enumerate(self.cars):
                leader_distance = None
                leader_velocity = None
                if car.car_id > 0:  # Assuming car_id 0 is the leader
                    leader_car = self.cars[car.car_id - 1]
                    leader_distance = leader_car.distance_travelled
                    leader_velocity = leader_car.velocity
                car.update(self.dt, leader_distance, leader_velocity)
            self.current_time += self.dt

    def save_logs_to_excel(self):
        """Save the logs of each car to an Excel file."""
        for car in self.cars:
            df = pd.DataFrame(car.logs, columns=['Time', 'Distance Travelled', 'Velocity', 'Control Velocity',
                                                 'Leader Distance', 'Leader Velocity'])
            df.to_excel(f'./tables/Car_{car.car_id}_{car.strategy}_{car.traffic_state}_logs.xlsx', index=False)


def create_cars(car_specs):
    cars = []
    for spec in car_specs:
        car = Car(
            car_id=spec['car_id'],
            initial_velocity=spec['initial_velocity'],
            initial_distance=spec['initial_distance'],
            max_velocity=spec['max_velocity'],
            max_acceleration=spec['max_acceleration'],
            min_acceleration=spec['min_acceleration'],
            desired_distance=spec['desired_distance'],
            desired_headway=spec['desired_headway'],
            strategy=spec['strategy']
        )
        cars.append(car)
    return cars

for model_used_ in ['ghr', 'gipps', 'idm', 'pid-v', 'acc']:

    car_specs = [
        {
            'car_id': 0,
            'initial_velocity': 0.0,
            'initial_distance': 1.5,
            'max_velocity': 1.6,
            'max_acceleration': 4.0,
            'min_acceleration': -6.0,
            'desired_distance': 0.5,
            'desired_headway': 0.5,
            'strategy': model_used_
        },
        {
            'car_id': 1,
            'initial_velocity': 0.0,
            'initial_distance': 1.0,
            'max_velocity': 1.6,
            'max_acceleration': 4.0,
            'min_acceleration': -6.0,
            'desired_distance': 0.5,
            'desired_headway': 0.5,
            'strategy': model_used_
        },
        {
            'car_id': 2,
            'initial_velocity': 0.0,
            'initial_distance': 0.5,
            'max_velocity': 1.6,
            'max_acceleration': 4.0,
            'min_acceleration': -6.0,
            'desired_distance': 0.5,
            'desired_headway': 0.5,
            'strategy': model_used_
        },
        {
            'car_id': 3,
            'initial_velocity': 0.0,
            'initial_distance': 0.0,
            'max_velocity': 1.6,
            'max_acceleration': 4.0,
            'min_acceleration': -6.0,
            'desired_distance': 0.5,
            'desired_headway': 0.5,
            'strategy': model_used_
        }
    ]
    
    # Create cars
    cars = create_cars(car_specs)
    # Simulation
    simulation = Simulation(cars=cars, simulation_time=60)
    simulation.run()
    simulation.save_logs_to_excel()
