
# %% [markdown]
# ## Installs and imports

# %%
# import torch

# %%
# !pip install networkx==3.1

# %%
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
# %pip install jupyter notebook

# %%
# system operations
import sys
# import matplotlib
print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("Matplotlib installed:", 'citylearn' in sys.modules)