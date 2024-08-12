import json
import requests
import logging
import paho.mqtt.client as mqtt

# Configuration for the MQTT broker
MQTT_BROKER = "34.72.65.111"

# MQTT_BROKER = "0"

MQTT_PORT = 1883
MQTT_TOPICS = [("esp32/consumption1", 0), ("esp32/consumption2", 0), ("esp32/consumption3", 0),("esp32/consumption4", 0),("esp32/consumption5", 0), ("esp32/sensors", 0)]

# Configuration for the PHP API
API_URL = "http://34.72.65.111/php_files/insert_consumption_data.php"  # Replace with your server's address if different
SENSOR_API_URL = "http://34.72.65.111/php_files/insert_sensor_data.php"  # API for sensor data

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# The callback for when the client receives a CONNACK response from the server
def on_connect(client, userdata, flags, rc):
    logging.info(f"Connected with result code {rc}")
    # Subscribing to topics
    for topic in MQTT_TOPICS:
        client.subscribe(topic)
        logging.info(f"Subscribed to {topic[0]}")

# The callback for when a PUBLISH message is received from the server
def on_message(client, userdata, msg):
    logging.debug(f"Message received from topic {msg.topic}: {msg.payload.decode()}")
    try:
        # Replace 'nan' with 'null' in the JSON string
        payload_str = msg.payload.decode().replace('nan', 'null')
        
        # Parse the JSON data from the MQTT message
        data = json.loads(payload_str)
        
        if msg.topic.startswith("esp32/consumption"):
            # Handle consumption data
            payload = {
                'DeviceID': data['deviceID'],
                'voltage': data['voltage'],
                'current': data['current'],
                'power': data['power'],
                'energy': data['energy']
            }
            logging.debug(f"Consumption payload to be sent to API: {payload}")
            # Send the data to the PHP API
            response = requests.post(API_URL, json=payload)
            logging.info(f"Consumption data sent to API. Response: {response.text}")
        elif msg.topic == "esp32/sensors":
            # Handle sensor data
            payload = {
                'Temperature1': data['temperature'],
                'Humidity1': data['humidity'],
                'Light1': data['ldrValue1'],
                'Light2': data['ldrValue2'],
                'Motion1': data['motionDetected']
            }
            logging.debug(f"Sensor payload to be sent to API: {payload}")
            # Send the data to the PHP API
            response = requests.post(SENSOR_API_URL, json=payload)
            logging.info(f"Sensor data sent to API. Response: {response.text}")
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON: {e}")
    except requests.RequestException as e:
        logging.error(f"Error sending data to API: {e}")
    except KeyError as e:
        logging.error(f"Missing expected key in data: {e}")

def main():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(MQTT_BROKER, MQTT_PORT, 60)

    # Blocking call that processes network traffic, dispatches callbacks, and handles reconnecting
    client.loop_forever()

if __name__ == "__main__":
    main()