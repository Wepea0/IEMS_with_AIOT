import gymnasium as gym
import numpy as np

import sinergym

new_time_variables=['month', 'day_of_month', 'hour']

new_variables = {
        "Site Outdoor Air DryBulb Temperature"                            : {
            "variable_names" : "outdoor_temperature",
            "keys"          : "Environment"           
        },
        "Site Outdoor Air Relative Humidity"                              : {
            "variable_names" : "outdoor_humidity",
            "keys"          : "Environment"         
        },
        "Site Wind Speed"                                                 : {
            "variable_names" : "wind_speed",
            "keys"          : "Environment"            
        },
        "Site Wind Direction"                                             : {
            "variable_names" : "wind_direction",
            "keys"          : "Environment"            
        },
        "Site Diffuse Solar Radiation Rate per Area"                      : {
            "variable_names" : "diffuse_solar_radiation",
            "keys"          : "Environment"         
        },
        "Site Direct Solar Radiation Rate per Area"                       : {
            "variable_names" : "direct_solar_radiation",
            "keys"          : "Environment"            
        },
        "Zone Thermostat Heating Setpoint Temperature"                    : {
            "variable_names" : "htg_setpoint",
            "keys"          : "SPACE5-1"            
        },
        "Zone Thermostat Cooling Setpoint Temperature"                    : {
            "variable_names" : "clg_setpoint",
            "keys"          : "SPACE5-1"           
        },
        "Zone Air Temperature"                                            : {
            "variable_names" : "air_temperature",
            "keys"          : "SPACE5-1"            
        },
        "Zone Air Relative Humidity"                                      : {
            "variable_names" : "air_humidity",
            "keys"          : "SPACE5-1"            
        },
        "Zone People Occupant Count"                                      : {
            "variable_names" : "people_occupant",
            "keys"          : "SPACE5-1"            
        },
        "Facility Total HVAC Electricity Demand Rate"                     : {
            "variable_names" : "HVAC_electricity_demand_rate",
            "keys"          : "Whole Building"            
        }
    }


new_meters = {
        "Electricity:HVAC" : "total_electricity_HVAC"
    }

new_actuators          = {
        "BASINHEATERSCHED"          : {
            "variable_name" : "Heating_Setpoint_RL",
            "element_type"  : "Schedule:Compact", 
            "value_type"    : "Schedule Value"
        },
        "AIRVELOCITYSCH"          : {
            "variable_name" : "Cooling_Setpoint_RL",
            "element_type"  : "Schedule:Compact", 
            "value_type"    : "Schedule Value"
        }
    }
# Schedule:Compact,Schedule Value,CLG-SETP-SC
# Schedule:Compact,Schedule Value,HTG-SETP-SCH


# new_action_space = gym.spaces.Box(
#     low=np.array([20, 26], dtype=np.int32),
#     high=np.array([26, 30 ], dtype=np.int32),
#     shape=(2,),
#     dtype=np.float32)

action_space = {"gym.spaces.Box(low=np.array([12.0, 23.25], dtype=np.float32), high=np.array([23.25, 30.0], dtype=np.float32), shape=(2,), dtype=np.float32)"}


env = gym.make('Eplus-5zone-hot-discrete-v1',
                time_variables=new_time_variables,
                variables=new_variables,
                meters=new_meters,
                actuators=new_actuators,
                action_space=new_action_space
                # weather_files='GHA_AA_Tema.654730_TMYx.2007-2021.epw'
            )

print('New environment observation varibles (time variables + variables + meters): {}'.format(env.get_wrapper_attr('observation_variables')))
print('New environment action varibles (actuators): {}'.format(env.get_wrapper_attr('action_variables')))
for i in range(1):
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