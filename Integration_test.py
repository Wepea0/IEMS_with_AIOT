# %% [markdown]
# ## Installs and imports

# %%
# !import torch


# !pip install networkx==3.1


# !pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# %%
# !pip install scikit-learn

# %%
# !nohup jupyter lab --ip=0.0.0.0 --port=8888 --no-browser &

# %%
# !disown %1

# %%
# %%capture

# # the environment we will be working with and its dependencies
# !pip install gymnasium
# !pip install nrel-pysam
# !pip install simplejson
# # TODO: update to install stable version from PyPi
# !pip install CityLearn

# # to generate static figures
# !pip install matplotlib
# !pip install seaborn

# # provide standard RL algorithms
# !pip install --no-deps stable-baselines3

# # results submission
# !pip install requests
# !pip install beautifulsoup4

# # progress bar
# !pip install tqdm

# #Python Sql connector
# !pip install pymysql

# #Micropython requests module
# !pip install urequests

# %%
# !pip install jupyter notebook




# %%

# system operations
#!/usr/bin/env python3.8

# import subprocess

# # Run the command and capture the output
# result = subprocess.run(['which', 'python3'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# # Get the path from the output
# python_path = result.stdout.strip()

# # Print the path
# print(f"Python executable path: {python_path}")




import inspect
import os
import uuid
import warnings
from typing import List, Tuple

# date and time
from datetime import datetime

# type hinting
from typing import Any

# User interaction
# from ipywidgets import Button, HTML
# from ipywidgets import Text, HBox, VBox

# data visualization
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from tqdm import tqdm

# data manipulation
import math
import numpy as np
import pandas as pd
import random
import re
import requests
import simplejson as json

# cityLearn
from citylearn.agents.base import (
    BaselineAgent,
)
from citylearn.agents.q_learning import TabularQLearning
from citylearn.citylearn import CityLearnEnv
from citylearn.data import DataSet
from citylearn.agents.rbc import HourRBC
from citylearn.data import Weather
from citylearn.reward_function import RewardFunction
from citylearn.wrappers import (
    NormalizedObservationWrapper,
    StableBaselines3Wrapper,
)
# from citylearn.py import CityLearnEnv
# RL algorithms
from stable_baselines3 import SAC

# %%
# set all plotted figures without margins
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0
# %matplotlib inline
import sys

print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("Matplotlib installed:", 'matplotlib' in sys.modules)


# %% [markdown]
# ## Setup environment 

# %%
selected_building = ["Building_1"]


# %%
DATASET_NAME = 'citylearn_challenge_2023_phase_1'
schema = DataSet.get_schema(DATASET_NAME)
root_directory = schema['root_directory']


# %%
'''View PRICING data'''
# change the suffix number in the next code line to a
# number between 1 and 17 to preview other buildings
selected_building = "Building_1"
filename = schema['buildings'][selected_building]['pricing']
filepath = os.path.join(root_directory, filename)
pricing_data = pd.read_csv(filepath)

# %% [markdown]
# 

# %%
# DATASET_NAME = 'citylearn_challenge_2022_phase_1'
# schemao = DataSet.get_schema(DATASET_NAME)
# root_directory = schemao['root_directory']
# # schemao

# %% [markdown]
# #### Weather File Wrangling
# 

# %%
'''View weather data'''
filename = schema['buildings'][selected_building]['weather']
filepath = os.path.join(root_directory, filename)
weather_data = pd.read_csv(filepath)
# building_data.info()

# for index, value in weather_data["outdoor_dry_bulb_temperature"] .items():
#     weather_data.at[index, 'outdoor_dry_bulb_temperature'] = round(random.uniform(24, 30), 2)

columns_to_reset = [
    "diffuse_solar_irradiance_predicted_6h", "diffuse_solar_irradiance_predicted_12h", 
    "diffuse_solar_irradiance_predicted_24h", "direct_solar_irradiance_predicted_6h",
    "direct_solar_irradiance_predicted_24h", "direct_solar_irradiance_predicted_12h",
    "outdoor_relative_humidity_predicted_24h", "outdoor_relative_humidity_predicted_12h", 
    "outdoor_relative_humidity_predicted_6h", "outdoor_dry_bulb_temperature_predicted_24h",
    "outdoor_dry_bulb_temperature_predicted_6h", "outdoor_dry_bulb_temperature_predicted_12h"
]

# Set all rows in the specified columns to 0
weather_data[columns_to_reset] = 0




# display(weather_data.head(10))






# %%
outdoor_temp = weather_data["outdoor_dry_bulb_temperature"].describe()


# %% [markdown]
# #### Energy simulation file wrangling

# %%
'''View energy simulation data'''
import random

filename = schema['buildings'][selected_building]['energy_simulation']
filepath = os.path.join(root_directory, filename)
building_data = pd.read_csv(filepath)
# display(building_data.head(10))

#Modify data to be more relevant
# building_data.loc[:, 'indoor_dry_bulb_temperature_set_point'] = 25
building_data.loc[:, 'hvac_mode'] = 3
# building_data.loc[:, 'average_unmet_cooling_setpoint_difference'] = 0

# building_data.loc[:, 'cooling_demand'] = 0
# building_data.loc[:, 'heating_demand'] = 0






for index, value in building_data["indoor_dry_bulb_temperature"] .items():
    building_data.at[index, 'indoor_dry_bulb_temperature'] = random.uniform(24, 30)


for index, value in building_data["indoor_dry_bulb_temperature_set_point"] .items():
    building_data.at[index, 'indoor_dry_bulb_temperature_set_point'] = random.uniform(24, 26)



# display(building_data.head(10))

# building_data.info()

# %%
indoor_temp = building_data["indoor_dry_bulb_temperature"].describe()



# %%
demand_comp = building_data[building_data["heating_demand"] > building_data["cooling_demand"]]
# print(demand_comp.info())

# %% [markdown]
#  cooling_or_heating_device_action - Fraction of cooling_device or heating_device nominal_power to make available. An action < 0.0 is for the cooling_device, while an action > 0.0 is for the heating_device.
#  

# %% [markdown]
# #### Editing heating demand

# %%
# def add_random_to_cooling_demand(row):
#     return row['cooling_demand'] + random.uniform(0.1, 1)

# # Apply the function to rows that meet the condition
# building_data.loc[building_data['indoor_dry_bulb_temperature'] < building_data['indoor_dry_bulb_temperature_set_point'], 'heating_demand'] = building_data[building_data['indoor_dry_bulb_temperature'] < building_data['indoor_dry_bulb_temperature_set_point']].apply(add_random_to_cooling_demand, axis=1)
# building_data.loc[building_data['indoor_dry_bulb_temperature'] < building_data['indoor_dry_bulb_temperature_set_point'], 'cooling_demand'] = 0


# building_data.loc[building_data['indoor_dry_bulb_temperature'] < building_data['indoor_dry_bulb_temperature_set_point'], 'heating_demand'] = building_data['cooling_demand'] 
# building_data.loc[building_data['heating_demand'] > 0, 'cooling_demand'] = 0



# building_data

# %%
demand_comp = building_data[building_data["indoor_dry_bulb_temperature"] < building_data["indoor_dry_bulb_temperature_set_point"]]

# demand_comp = building_data[building_data["heating_demand"] > building_data["cooling_demand"]]
# print(demand_comp.info())

# %%
building_data.head(20)


# %%
occupant_info = building_data["heating_demand"].describe()
occupant_info

# %%
occupant_info = building_data["cooling_demand"].describe()
occupant_info


# %% [markdown]
# #### Graphs

# %%
'''Plot non-shiftable load vs solar generation'''

fig, axs = plt.subplots(1, 2, figsize=(14, 2.5))
x = building_data.index
y1 = building_data['non_shiftable_load'] 
y2 = building_data['solar_generation']
axs[0].plot(x, y1)
axs[0].set_xlabel('Time step')
axs[0].set_ylabel('Non-shiftable load\n[kWh]')
axs[1].plot(x, y2)
axs[1].set_xlabel('Time step')
axs[1].set_ylabel('Solar generation\n[W/kW]')
fig.suptitle(selected_building)
plt.tight_layout()
plt.show()

# %%
weather_data

# %%
'''Plot total load vs solar generation'''

fig, axs = plt.subplots(1, 2, figsize=(14, 2.5))
x = building_data.index
y1 = building_data['non_shiftable_load'] + building_data["cooling_demand"]
y2 = building_data['solar_generation']
axs[0].plot(x, y1)
axs[0].set_xlabel('Time step')
axs[0].set_ylabel('Total load\n[kWh]')
axs[1].plot(x, y2)
axs[1].set_xlabel('Time step')
axs[1].set_ylabel('Solar generation\n[W/kW]')
fig.suptitle(selected_building)
plt.tight_layout()
plt.show()

# %%
weather_data

# %%
'''Plot weather data'''
columns = [
    'outdoor_dry_bulb_temperature', 'outdoor_relative_humidity',
    'diffuse_solar_irradiance', 'direct_solar_irradiance'
]
titles = [
    'Outdoor dry-bulb\ntemperature [C]', 'Relative humidity\n[%]',
    'Diffuse solar irradiance\n[W/m2]', 'Direct solar irradiance\n[W/m2]'
]
fig, axs = plt.subplots(2, 2, figsize=(14, 4.25))
x = weather_data.index

for ax, c, t in zip(fig.axes, columns, titles):
    y = weather_data[c]
    ax.plot(x, y)
    ax.set_xlabel('Time step')
    ax.set_ylabel(t)

fig.align_ylabels()
plt.tight_layout()
plt.show()

# %%
weather_data.head()

# %%

'''View carbon intensity data'''
filename = schema['buildings'][selected_building]['carbon_intensity']
filepath = os.path.join(root_directory, filename)
carbon_intensity_data = pd.read_csv(filepath)
# display(carbon_intensity_data.head())
# carbon_intensity_data.info()

# %% [markdown]
# ### Save changes to file
# 

# %%
filename = schema['buildings'][selected_building]['energy_simulation']
filepath = os.path.join(root_directory, filename)
building_data.to_csv(filepath, index=False)

filename = schema['buildings'][selected_building]['weather']
filepath = os.path.join(root_directory, filename)
weather_data.to_csv(filepath, index=False)

# %%
building_data

# %%
filename = schema['buildings'][selected_building]['energy_simulation']
filepath = os.path.join(root_directory, filename)
building_data = pd.read_csv(filepath)
building_data.head(24)

# %%
weather_data

# %% [markdown]
# ### Graph stuff

# %%
grid_electricity_consumed = 0
baseline_cost = 0 # Baseline Cost, Agent Cost
agent_cost = 0
old_pricing_cost = 0

def get_kpis(env: CityLearnEnv) -> pd.DataFrame:
    """Returns evaluation KPIs.

    Electricity cost and carbon emissions KPIs are provided
    at the building-level and average district-level. Average daily peak,
    ramping and (1 - load factor) KPIs are provided at the district level.

    Parameters
    ----------
    env: CityLearnEnv
        CityLearn environment instance.

    Returns
    -------
    kpis: pd.DataFrame
        KPI table.
    """

    kpis = env.unwrapped.evaluate()

    # names of KPIs to retrieve from evaluate function
    kpi_names = {
        'cost_total': 'Cost',
        'carbon_emissions_total': 'Emissions',
        'discomfort_cold_proportion': 'Cold discomfort proportion',
        'electricity_consumption_total': 'Grid electricity consumed',
        'daily_peak_average': 'Daily Peak Average',
        'old_pricing_cost':'Old pricing system cost'
       
        # 'one_minus_thermal_resilience_proportion': 'Thermal resilience proportion'
    }


    kpis = kpis[
        (kpis['cost_function'].isin(kpi_names))
    ].dropna()
    kpis['cost_function'] = kpis['cost_function'].map(lambda x: kpi_names[x])

    # round up the values to 2 decimal places for readability
    kpis['value'] = kpis['value'].round(2)

    # rename the column that defines the KPIs
    kpis = kpis.rename(columns={'cost_function': 'kpi'})


    return kpis

# %%
def plot_building_kpis(envs) -> plt.Figure:
    """Plots electricity consumption, cost, and carbon emissions
    at the building-level for different control agents in bar charts.

    Parameters
    ----------
    envs: dict[str, CityLearnEnv]
        Mapping of user-defined control agent names to environments
        the agents have been used to control.

    Returns
    -------
    fig: plt.Figure
        Figure containing plotted axes.
    """
    global old_pricing_cost 
    global agent_cost 
    global baseline_cost 


    kpis_list = []

    for k, v in envs.items():
        kpis = get_kpis(v)
        # kpis = kpis[kpis['level']=='building'].copy()
        kpis['building_id'] = kpis['name'].str.split('_', expand=True)[1]
        kpis['building_id'] = kpis['building_id'].astype(int).astype(str)
        kpis['env_id'] = k
        kpis_list.append(kpis)
        # print("KPIS LIST", kpis_list)
        
    # print(kpis_list[1].keys())
    # print(kpis_list[1])
    # grid_electricity_consumed = kpis_list[1]['value'][17]
    baseline_cost = kpis_list[0]['value'][3]
    agent_cost = kpis_list[1]['value'][3]
    old_pricing_cost = kpis_list[1]['value'][17]

    

    # print("Baseline cost - ", baseline_cost)
    # print("Agent cost -", agent_cost)
    # print("Old pricing cost ", old_pricing_cost)

    # kpis_list[1].pop(17)
    # kpis_list[0].pop(17)


    

    # print(" *********** KPIS ", kpis_list)
    
            


    kpis = pd.concat(kpis_list, ignore_index=True, sort=False)
    kpi_names = kpis['kpi'].unique()

    # Adjusted to accommodate 6 charts in a 2x3 grid
    column_count = 1
    row_count = 6
    figsize = (10, 8)  # Adjusted figure size for better visibility and spacing

    fig, axs = plt.subplots(row_count, column_count, figsize=figsize, sharey=True)

    

    for i, (ax, (k, k_data)) in enumerate(zip(axs.flatten(), kpis.groupby('kpi'))):
        sns.barplot(x='value', y='name', data=k_data, hue='env_id', ax=ax, width=0.6)  # Reduced bar width
        ax.set_title(k, fontsize=40)  # Increased font size of heading
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.set_title(k)

        for j, _ in enumerate(envs):
            ax.bar_label(ax.containers[j], fmt='%.2f')

        if i % column_count == 0:  # Only show legend for the first column of each row
            ax.legend(loc='upper left', bbox_to_anchor=(1.3, 1.0), framealpha=0.0)
        else:
            ax.legend().set_visible(False)

        for s in ['right', 'top']:
            ax.spines[s].set_visible(False)

    # plt.subplots_adjust(wspace=0.6, hspace=0.8) 
    plt.subplots_adjust(wspace=4.0, hspace=0.8) 


    return fig




# %%
def plot_building_load_profiles(
    envs, daily_average: bool = None
) -> plt.Figure:
    """Plots building-level net electricty consumption profile
    for different control agents.

    Parameters
    ----------
    envs: dict[str, CityLearnEnv]
        Mapping of user-defined control agent names to environments
        the agents have been used to control.
    daily_average: bool, default: False
        Whether to plot the daily average load profile.

    Returns
    -------
    fig: plt.Figure
        Figure containing plotted axes.
    """

    daily_average = False if daily_average is None else daily_average
    building_count = len(list(envs.values())[0].buildings)
    column_count_limit = 4
    row_count = math.ceil(building_count/column_count_limit)
    column_count = min(column_count_limit, building_count)
    figsize = (8.0*column_count, 3*row_count)
    fig, _ = plt.subplots(row_count, column_count, figsize=figsize)

    for i, ax in enumerate(fig.axes):
        for k, v in envs.items():
            y = v.unwrapped.buildings[i].net_electricity_consumption
            y = np.reshape(y, (-1, 24)).mean(axis=0) if daily_average else y
            x = range(len(y))
            ax.plot(x, y, label=k)

        ax.set_title(v.unwrapped.buildings[i].name)
        ax.set_ylabel('kWh')

        if daily_average:
            ax.set_xlabel('Hour')
            ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

        else:
            ax.set_xlabel('Time step')
            ax.xaxis.set_major_locator(ticker.MultipleLocator(24))

        if i == building_count - 1:
            ax.legend(
                loc='upper left', bbox_to_anchor=(1.0, 1.0), framealpha=0.0
            )
        else:
            ax.legend().set_visible(False)


    plt.tight_layout()

    return fig


# %%
def plot_battery_soc_profiles(envs) -> plt.Figure:
    """Plots building-level battery SoC profiles fro different control agents.

    Parameters
    ----------
    envs: dict[str, CityLearnEnv]
        Mapping of user-defined control agent names to environments
        the agents have been used to control.

    Returns
    -------
    fig: plt.Figure
        Figure containing plotted axes.
    """

    building_count = len(list(envs.values())[0].buildings)
    column_count_limit = 4
    row_count = math.ceil(building_count/column_count_limit)
    column_count = min(column_count_limit, building_count)
    figsize = (8.0*column_count, 3*row_count)
    fig, _ = plt.subplots(row_count, column_count, figsize=figsize)

    for i, ax in enumerate(fig.axes):
        for k, v in envs.items():
            y = np.array(v.unwrapped.buildings[i].electrical_storage.soc)
            x = range(len(y))
            ax.plot(x, y, label=k)

        ax.set_title(v.unwrapped.buildings[i].name)
        ax.set_xlabel('Time step')
        ax.set_ylabel('SoC')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(24))
        ax.set_ylim(0.0, 1.0)

        if i == building_count - 1:
            ax.legend(
                loc='upper left', bbox_to_anchor=(1.0, 1.0), framealpha=0.0
            )
        else:
            ax.legend().set_visible(False)


    plt.tight_layout()

    return fig

# %%
# def plot_simulation_summary(envs):
#     """Plots KPIs, load and battery SoC profiles for different control agents.

#     Parameters
#     ----------
#     envs: dict[str, CityLearnEnv]
#         Mapping of user-defined control agent names to environments
#         the agents have been used to control.
#     """

#     # print('#'*8 + ' BUILDING-LEVEL ' + '#'*8)
#     # print('Building-level KPIs:')
#     _ = plot_building_kpis(envs)
#     plt.show()

#     # print('Building-level simulation period load profiles:')
#     _ = plot_building_load_profiles(envs)
#     plt.show()

#     # print('Building-level daily-average load profiles:')
#     _ = plot_building_load_profiles(envs, daily_average=True)
#     plt.show()

#     # print('Battery SoC profiles:')
#     _ = plot_battery_soc_profiles(envs)
#     plt.show()



# %% [markdown]
# ### Pre-model setup steps

# %%
'''Randomly select period of simulation'''
def select_simulation_period(
    dataset_name: str, count: int, seed: int, simulation_periods_to_exclude: List[Tuple[int, int]] = None) -> Tuple[int, int]:
    """Randomly select environment simulation start and end time steps
    that cover a specified number of days.

    Parameters
    ----------
    dataset_name: str
        CityLearn dataset to query buildings from.
    count: int
        Number of simulation days.
    seed: int
        Seed for pseudo-random number generator.
    simulation_periods_to_exclude: list[tuple[int, int]]
        List of simulation periods to exclude from selection pool.

    Returns
    -------
    simulation_start_time_step: int
        The first time step in schema time series files to
        be read when constructing the environment.
    simulation_end_time_step: int
        The last time step in schema time series files to
        be read when constructing the environment.
    """

    assert 1 <= count <= 365, 'count must be between 1 and 365.'

    # set random seed
    np.random.seed(seed)

    # use any of the files to determine the total
    # number of available time steps
    schema = DataSet.get_schema(dataset_name)
    filename = schema['buildings'][selected_building]['carbon_intensity']
    filepath = os.path.join(root_directory, filename)
    time_steps = pd.read_csv(filepath).shape[0]

    # set candidate simulation start time steps
    # spaced by the number of specified days
    simulation_start_time_step_list = np.arange(0, time_steps, 24*count)

    # exclude period if needed
    if simulation_periods_to_exclude is not None:
        simulation_start_time_step_list_to_exclude = \
            [s for s, e in simulation_periods_to_exclude]
        simulation_start_time_step_list = np.setdiff1d(
            simulation_start_time_step_list,
            simulation_start_time_step_list_to_exclude
        )

    else:
        pass

    # randomly select a simulation start time step
    simulation_start_time_step = np.random.choice(
        simulation_start_time_step_list, size=1
    )[0]
    simulation_end_time_step = simulation_start_time_step + 24*count - 1

    return simulation_start_time_step, simulation_end_time_step

# %%
random_seed = 7


# %%
'''Pick simulation time steps'''
simulation_start, simulation_end = select_simulation_period(DATASET_NAME, 30, random_seed)
simulation_start

# %% [markdown]
# average_unmet_cooling_setpoint_difference - Difference between indoor_dry_bulb_temperature and cooling temperature 

# %%
'''Pick observations and agent'''
# active_observations = ["hour", "indoor_dry_bulb_temperature", "electricity_pricing", "indoor_dry_bulb_temperature_delta", "non_shiftable_load", "average_unmet_cooling_setpoint_difference",]
# active_observations = ["hour", "indoor_dry_bulb_temperature", "electricity_pricing",  
#                        "non_shiftable_load", "solar_generation" 
#                         "outdoor_dry_bulb_temperature"]

active_observations = ["hour", "indoor_dry_bulb_temperature", "electricity_pricing", 
                       "non_shiftable_load", "solar_generation", "indoor_dry_bulb_temperature_set_point", 
                         ]


CENTRAL_AGENT = True

# %% [markdown]
# ## Model Inference

# %% [markdown]
# #### Access temporal data

# %%
from datetime import datetime

curr_month = datetime.now().month
curr_day = datetime.weekday(datetime.today()) + 1
curr_hour = datetime.now().hour


# %% [markdown]
# #### Open data files

# %%
loaded_sac_model = SAC.load("/home/wepea2/capstone/sinergym/drl_scripts/Outage_final_model.zip")

# %%
filename = schema['buildings'][selected_building]['energy_simulation']
filepath = os.path.join(root_directory, filename)
energy_simulation = pd.read_csv(filepath)
energy_simulation.head()

# %%
#Get applicable path on raspberry pi
TOU_FILEPATH = "/home/wepea2/capstone/sinergym/drl_scripts/TOU_pricing.csv"

pricing = pd.read_csv(TOU_FILEPATH)
pricing = pricing[:24]
pricing = pd.concat([pricing] * 30, ignore_index=True)
pricing


# %%
# curr_price_info = pricing.iloc[:1].copy()
# curr_pricing_range = pricing.loc[curr_hour].copy()
# curr_price_info["electricity_pricing"] = curr_pricing_range["electricity_pricing"]
# curr_price_info["electricity_pricing_predicted_6h"] = curr_pricing_range["electricity_pricing_predicted_6h"]
# curr_price_info["electricity_pricing_predicted_12h"] = curr_pricing_range["electricity_pricing_predicted_12h"]
# curr_price_info["electricity_pricing_predicted_24h"] = curr_pricing_range["electricity_pricing_predicted_24h"]

# curr_price_info


# %%
#Save pricing information
filename = schema['buildings'][selected_building]['pricing']
filepath = os.path.join(root_directory, filename)
pricing.to_csv(filepath, index=False)

# %%


# %%
filename = schema['buildings'][selected_building]['weather']
filepath = os.path.join(root_directory, filename)
weather_data = pd.read_csv(filepath)


# weather_data = weather_data[:1]
weather_data

# %% [markdown]
# #### Access weather API data

# %%
#Access weather data
import requests
import time 
import json

def make_api_call(url, headers=None, data=None, retries=3, backoff_factor=1):
    for attempt in range(retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"API call failed with error {response.status_code}")

        except Exception as err:
            print(f"API call failed: {err}")
            if attempt < retries -1:
                time.sleep(backoff_factor * (2 ** attempt))
            else:
                raise "Failed. Max retries exceeded"
            

api_url = r"https://student_chubs_chubbylina:5S89jIlaEh@api.meteomatics.com"




# %%
LOCATION = "5.7348,0.0302" #Tema
DATE_SUFFIX = "ZPT1H"
TEMPERATURE_CODE = "t_2m:C"
HUMIDITY_CODE = "relative_humidity_2m:p"
DIFFUSE_IRRADIANCE_CODE = "diffuse_rad:W"
DIRECT_IRRADIANCE_CODE = "direct_rad:W"


curr_date_time = datetime.now()
curr_date_time = curr_date_time.isoformat()
curr_date_time = curr_date_time + DATE_SUFFIX
# print(new_date)

# https://student_buntugu_wepea:ps9a4CjKR9@api.meteomatics.com/2024-07-27T17:53:30.035099ZPT1H/t_2m:C/5.7348,0.0302/json
# temperature_url = api_url + "/" + curr_date_time +  "/" + TEMPERATURE_CODE + "/" + LOCATION + "/json"

def get_temperature(curr_date_time):
    temperature_url = api_url + "/" + curr_date_time +  "/" + TEMPERATURE_CODE + "/" + LOCATION + "/json"
    response = make_api_call(temperature_url)
    # print("API response -", response)
    return response

def get_humidity(curr_date_time):
    humidity_url = api_url + "/" + curr_date_time +  "/" + HUMIDITY_CODE + "/" + LOCATION + "/json"
    response = make_api_call(humidity_url)
    # print("API response -", response)
    return response

def get_diffuse_solar_irradiance(curr_date_time):
    diffuse_irradiance_url = api_url + "/" + curr_date_time +  "/" + DIFFUSE_IRRADIANCE_CODE + "/" + LOCATION + "/json"
    response = make_api_call(diffuse_irradiance_url)
    # print("API response -", response)
    return response

def get_direct_solar_irradiance(curr_date_time):
    direct_irradiance_url = api_url + "/" + curr_date_time +  "/" + DIRECT_IRRADIANCE_CODE + "/" + LOCATION + "/json"
    response = make_api_call(direct_irradiance_url)
    # print("API response -", response)
    return response


# %%
#Get temperature
temperature_api_response = get_temperature(curr_date_time)
temperature_value = temperature_api_response['data'][0]['coordinates'][0]['dates'][0]['value']
print(temperature_value)

#Get humidity
humidity_api_response = get_humidity(curr_date_time)
humidity_value = humidity_api_response['data'][0]['coordinates'][0]['dates'][0]['value']
print(humidity_value)

#Get diffuse solar irradiance
diffuse_irradiance_api_response = get_diffuse_solar_irradiance(curr_date_time)
diffuse_irradiance_value = diffuse_irradiance_api_response['data'][0]['coordinates'][0]['dates'][0]['value']
print(diffuse_irradiance_value)

#Get direct solar irradiance
direct_irradiance_api_response = get_direct_solar_irradiance(curr_date_time)
direct_irradiance_value = direct_irradiance_api_response['data'][0]['coordinates'][0]['dates'][0]['value']
print(direct_irradiance_value)

# %%
# direct_irradiance_value = value = direct_irradiance_api_response['data'][0]['coordinates'][0]['dates'][0]['value']
# print(direct_irradiance_value)

# %%
weather_data.head()
weather_data["outdoor_dry_bulb_temperature"] = temperature_value
weather_data["outdoor_relative_humidity"] = humidity_value
weather_data["diffuse_solar_irradiance"] = diffuse_irradiance_value
weather_data["direct_solar_irradiance"] = direct_irradiance_value



# %%
#Save weather data
filename = schema['buildings'][selected_building]['weather']
filepath = os.path.join(root_directory, filename)
weather_data.to_csv(filepath, index=False)

# %%
weather_data

# %% [markdown]
# ### Energy simulation inference file creation

# %%
energy_simulation

# %% [markdown]
# #### Get sensor data from database

# %%
latest_non_shiftable_load = 0

import pymysql
# Database connection
connection = pymysql.connect(
    host='localhost',
    user='vboxuser',
    password='vboxuser',
    database='House_environment'
)

try:
    with connection.cursor() as cursor:
        # Query for the latest temperature and humidity
        query_sensordata = "SELECT Temperature1, Humidity1 FROM sensordata ORDER BY timestamp DESC LIMIT 1"
        cursor.execute(query_sensordata)
        
        # Fetch the result for sensor data
        result_sensordata = cursor.fetchone()
        if result_sensordata:
            latest_temperature = result_sensordata[0]
            latest_humidity = result_sensordata[1]
            print(f"Latest Temperature1: {latest_temperature}, Latest Humidity1: {latest_humidity}")
        
        # Query for the most recent voltage and current readings for DeviceID 1 and 3
        query_consumptiondata = """
        SELECT DeviceID, Power
        FROM consumptiondata
        WHERE DeviceID IN (1, 3)
        ORDER BY timestamp DESC
        LIMIT 2
        """
        cursor.execute(query_consumptiondata)
        
        # Fetch the results for consumption data
        results_consumptiondata = cursor.fetchall()
        latest_non_shiftable_load = 0
        for row in results_consumptiondata:
            device_id = row[0]
            # voltage = row[1]
            # current = row[2]
            latest_non_shiftable_load += row[1]
            # try:
            #     power = voltage * current 
            #     latest_non_shiftable_load += power
            # finally:
            #     power = 0.3468944773948613/2
            #     latest_non_shiftable_load += power

            
        print(f"DeviceID {device_id} - Power: {latest_non_shiftable_load}")
        
        # Commit the transaction
        connection.commit()
finally:
    connection.close()


# %%
energy_simulation_sliced = energy_simulation.iloc[:].copy()
energy_simulation_sliced["month"] = curr_month
energy_simulation_sliced["day_type"] = curr_day
energy_simulation_sliced["hour"] = curr_hour

energy_simulation_sliced["indoor_dry_bulb_temperature"] = latest_temperature
energy_simulation_sliced["indoor_relative_humidity"] = latest_humidity
energy_simulation_sliced["indoor_dry_bulb_temperature_set_point"] = 25 #TODO Make this an if (to first check for setpoint in db)
energy_simulation_sliced["hvac_mode"] = 2
energy_simulation_sliced["dhw_demand"]  =random.uniform(1.0, 3.0) 
energy_simulation_sliced["average_unmet_cooling_setpoint_difference"] = energy_simulation_sliced["indoor_dry_bulb_temperature"] - energy_simulation_sliced["indoor_dry_bulb_temperature_set_point"] 
# energy_simulation_sliced["cooling_demand"] =  


# energy_simulation_sliced["solar_generation"] = 


energy_simulation_sliced["non_shiftable_load"] = latest_non_shiftable_load
occupant_count = 3
import random
random.seed(time.time())

# Generate a random floating-point number between 0 and 3
random_number = random.randint(0, 5)

print("ORIGIN OCC ",  random_number)
energy_simulation_sliced["occupant_count"]


# %%
filename = schema['buildings'][selected_building]['energy_simulation']
filepath = os.path.join(root_directory, filename)
energy_simulation_sliced.to_csv(filepath, index=False)


# %%
def plot_loaded_simulation_summary(envs):
    """Plots KPIs, load and battery SoC profiles for different control agents.

    Parameters
    ----------
    envs: dict[str, CityLearnEnv]
        Mapping of user-defined control agent names to environments
        the agents have been used to control.
    """

    print('#'*8 + ' BUILDING-LEVEL ' + '#'*8)
    print('Building-level KPIs:')
    _ = plot_building_kpis(envs)
    plt.show()

    # print('Building-level daily-average load profiles:')
    # _ = plot_building_load_profiles(envs, daily_average=True)
    # plt.show()




# %% [markdown]
# ### API activities

# %% [markdown]
# ### Get model actions

# %%

simulation_start = random.randint(0, 718)

loaded_baseline_env = CityLearnEnv(
    DATASET_NAME,
    central_agent=CENTRAL_AGENT,
    buildings=selected_building,
    active_observations=active_observations,
    simulation_start_time_step=simulation_start,
    simulation_end_time_step=simulation_start+1,
)

loaded_baseline_model = BaselineAgent(loaded_baseline_env)

# always start by reseting the environment
observations, _ = loaded_baseline_env.reset()

# step through the environment until terminal
# state is reached i.e., the control episode ends
while not loaded_baseline_env.terminated:
    # select actions from the model
    actions = loaded_baseline_model.predict(observations)

    # apply selected actions to the environment
    observations, _, _, _, _ = loaded_baseline_env.step(actions)

# %%
# plot_simulation_summary({
#     'Baseline': loaded_baseline_env,
# })

# %%
loaded_sac_env = CityLearnEnv(
    DATASET_NAME,
    central_agent=CENTRAL_AGENT,
    buildings=selected_building,
    active_observations=active_observations,
    simulation_start_time_step=simulation_start,
    simulation_end_time_step=simulation_start+1,
)

loaded_sac_env = StableBaselines3Wrapper(loaded_sac_env)

# loaded_sac_model = SAC(policy='MlpPolicy', env=loaded_sac_env, seed=random_seed)

observations, _ = loaded_sac_env.reset()
loaded_sac_actions_list = []

while not loaded_sac_env.unwrapped.terminated:
    actions, _ = loaded_sac_model.predict(observations, deterministic=True)
    observations, _, _, _, _ = loaded_sac_env.step(actions)
    loaded_sac_actions_list.append(actions)




# %%
# plot summary and compare with other control results
plot_loaded_simulation_summary({
    'Baseline': loaded_baseline_env,
    'SAC-1': loaded_sac_env
})

# %%
#Get inference actions
print(loaded_sac_actions_list)

# %%
import datetime

current_date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:00")

print(current_date_time)


# %%
cooling_action_list = []
ess_action_list = []
final_action_list = []
for i in loaded_sac_actions_list:
    for action, action_2 in np.array_split(i, 1):
        # print(action, " || ", action_2)
        action_tuple = (action, action_2, current_date_time, baseline_cost, agent_cost, old_pricing_cost)
        # final_action_list.append(action)
        # final_action_list.append(action_2)
        final_action_list.append(action_tuple)
# print("Cooling action list - ", cooling_action_list)
# print("ESS actions list = ", ess_action_list)
# print(final_action_list)







# %%
import datetime
timestamp = 1684929490
date_time = datetime.datetime.fromtimestamp(timestamp)
date_time_2 =  datetime.datetime.fromtimestamp(1684929540)
date_time_3 =  datetime.datetime.fromtimestamp(1684951200)
print(date_time)
print(date_time_2)
print(date_time_3)

# %%


# %%
import pymysql
# Database connection
connection = pymysql.connect(
    host='localhost',
    user='vboxuser',
    password='vboxuser',
    database='House_environment'
)

try:
    with connection.cursor() as cursor:
        # Construct the SQL query
        query = "INSERT INTO model_actions (cooling_action, ess_action, timestep, Baseline_cost, Agent_cost, Old_pricing_cost) VALUES (%s, %s, %s, %s, %s, %s)"
        
        # Execute the query for each tuple in the data list
        cursor.executemany(query, final_action_list)


        
        # Commit the transaction
        connection.commit()
finally:
    connection.close()


action = final_action_list[0][0]

light_on = 0
fan_on = 0
light_on = 0
light_on_marker = random.randint(0, 3)
if(light_on_marker > 1):
    light_on = 1


if(random_number > 3):
    fan_on = 1

print("action light - ", light_on_marker)
print("ACTION (for fan) - ", random_number)

print("\nActual action ", action)
print("Actual occupancy - ", energy_simulation_sliced["occupant_count"][simulation_start])

device_control_list = []
device_control_list.append(light_on)
device_control_list.append(fan_on)
device_control_list.append(light_on)
print(device_control_list)


import pymysql
# Database connection
connection = pymysql.connect(
    host='localhost',
    user='vboxuser',
    password='vboxuser',
    database='House_environment'
)

try:
    with connection.cursor() as cursor:
        # Construct the SQL query
        query = "UPDATE device_status SET device1 = %s, device2 = %s, device3 = %s"
        
        # Execute the query for each tuple in the data list
        cursor.execute(query, tuple(device_control_list))
        connection.commit()
       
        
        # Commit the transaction
        connection.commit()
finally:
    connection.close()



print("\n***********************\n")

