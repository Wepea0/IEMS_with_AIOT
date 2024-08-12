# CityLearn Simulator and Predictor

This project includes a CityLearn simulation and prediction system. It is designed to work with MQTT for data communication and requires weather data from Meteomatics via an API key.

## Prerequisites

Before running the simulator and predictor, ensure that the following software and tools are installed on your system:

- Python 3.x
- Required Python packages (install via `requirements.txt`)
- MQTT broker
- Web server (e.g., Apache) with PHP support
- MySQL or compatible database

1. **Obtain Meteomatics API Key**: Before starting, ensure you have a valid API key from Meteomatics. This key is required for accessing weather data.

2. **Run MQTT Broker**: Prior to executing the `Integration_test.py`, you must run the `mqtt_broker.py`. This script sets up the MQTT broker necessary for communication between components.

   During the execution of `mqtt_broker.py`, replace the placeholder for the database IP address with the actual IP address of the device hosting your database.

3. **Update PHP Files**: After starting the MQTT broker, update the `insert_consumption_data.php` and `insert_sensor_data.php` files located at `/var/www/html`. Replace the IP address placeholders in these files with the IP address of the machine running `mqtt_broker.py`.

4. **Execute Integration Test**: With the MQTT broker running and PHP files updated, proceed to execute the `Integration_test.py`.


   Ensure you replace the placeholder for the Meteomatics API key within `Integration_test.py` with your actual API key obtained in step 1.


Do note that default Citylearn installation will not produce the same results as in the experiment, the upload of the custom Citylearn package is still being worked on.

By following these steps, you will have successfully set up and executed the CityLearn Simulator and Predictor along with the necessary MQTT broker and PHP scripts for data insertion. Remember to check the logs for any errors and ensure all components are communicating as expected.
