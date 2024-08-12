import gymnasium as gym
import numpy as np

import sinergym

new_time_variables=['month', 'day_of_month', 'hour']

new_variables={
    'outdoor_temperature': ('Site Outdoor Air Drybulb Temperature', 'Environment'),
    'outdoor_humidity': ('Site Outdoor Air Relative Humidity', 'Environment'),
    'wind_speed': ('Site Wind Speed', 'Environment'),
    'wind_direction': ('Site Wind Direction', 'Environment'),
    'diffuse_solar_radiation': ('Site Diffuse Solar Radiation Rate per Area', 'Environment'),
    'direct_solar_radiation': ('Site Direct Solar Radiation Rate per Area', 'Environment'),
    'west_zone_htg_setpoint': ('Zone Thermostat Heating Setpoint Temperature', 'West Zone'),
    'east_zone_htg_setpoint': ('Zone Thermostat Heating Setpoint Temperature', 'East Zone'),
    'west_zone_clg_setpoint': ('Zone Thermostat Cooling Setpoint Temperature', 'West Zone'),
    'east_zone_clg_setpoint': ('Zone Thermostat Cooling Setpoint Temperature', 'East Zone'),
    'west_zone_air_temperature': ('Zone Air Temperature', 'West Zone'),
    'east_zone_air_temperature': ('Zone Air Temperature', 'East Zone'),
    'west_zone_air_humidity': ('Zone Air Relative Humidity', 'West Zone'),
    'east_zone_air_humidity': ('Zone Air Relative Humidity', 'East Zone'),
    'HVAC_electricity_demand_rate': ('Facility Total HVAC Electricity Demand Rate', 'Whole Building')
}

new_meters={
    'east_zone_electricity':'Electricity:Zone:EAST ZONE',
    'west_zone_electricity':'Electricity:Zone:WEST ZONE',
}

new_actuators = {
    'Heating_Setpoint_RL': (
        'Schedule:Compact',
        'Schedule Value',
        'Heating Setpoints'),
    'Cooling_Setpoint_RL': (
        'Schedule:Compact',
        'Schedule Value',
        'Cooling Setpoints')
}

new_action_space = gym.spaces.Box(
    low=np.array([14.0, 22.0], dtype=np.float32),
    high=np.array([22.0, 30.5], dtype=np.float32),
    shape=(2,),
    dtype=np.float32)

env = gym.make('Eplus-datacenter-cool-continuous-stochastic-v1',
                time_variables=new_time_variables,
                variables=new_variables,
                meters=new_meters,
                actuators=new_actuators,
                action_space=new_action_space,
            )

print('New environment observation varibles (time variables + variables + meters): {}'.format(env.get_wrapper_attr('observation_variables')))
print('New environment action varibles (actuators): {}'.format(env.get_wrapper_attr('action_variables')))
for i in range(5):
    obs, info = env.reset()
    rewards = []
    truncated = terminated = False
    current_month = 0
    while not (terminated or truncated):
        a = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(a)
        rewards.append(reward)
    print(
        'Episode ',
        i,
        'Mean reward: ',
        np.mean(rewards),
        'Cumulative reward: ',
        sum(rewards))
env.close()