#!/bin/bash

# # Path to your Python executable
# PYTHON_EXECUTABLE="/usr/bin/python3"

# # Path to your requirements.txt file
# REQUIREMENTS_FILE="/home/wepea2/capstone/sinergym/drl_scripts/requirements.txt"

# # Path to your Python script
# PYTHON_SCRIPT="/home/wepea2/capstone/sinergym/drl_scripts/Integration_test.py"

# # # Install the required modules
# # $PYTHON_EXECUTABLE -m pip install -r $REQUIREMENTS_FILE

# # Execute your Python script
# $PYTHON_EXECUTABLE $PYTHON_SCRIPT

#!/bin/bash

# Log file to store the output
LOG_FILE=/home/wepea2/capstone/sinergym/drl_scripts/cron.log

# Get the path of the Python executable
PYTHON_PATH=$(which python3)

# Log the Python executable path
echo "Python executable path: $PYTHON_PATH" >> $LOG_FILE