#!/usr/bin/env python
import logging
from azure.iot.device import IoTHubDeviceClient, Message
from azure.iot.device.exceptions import CredentialError, ConnectionFailedError, ConnectionDroppedError, ClientError

import config
import time


class MessageSender:

    image_path = None                   # Holds the image for mobile alert message

    client = None

    def __init__(self, config):

        # Set user configurations
        self.__conf_iothub_connection_str = config.get_iot_hub_conn_string()

    #function __iothub_client_init
    #Desctription: create a client from the specified connection string
    #Return value: result
    def __iothub_client_init(self):
        """create an iot hub client"""
        try:
            #create client from connection string
            if self.client is None:
                self.client = IoTHubDeviceClient.create_from_connection_string(self.__conf_iothub_connection_str)
            
        except Exception as ex:
            print("Unexpected error in connecting to client{0}".format(ex))
            logging.exception(ex)
            result = None

        else:
            result = self.client
        finally:
            pass

        return result

    #function iothib_client_send_message()
    #Description: function that sends message to azure hub
    #Parameter: time_now, person_id, location_x, location_y
    #Return value: True/False
    def iothub_client_send_message(self, msg_txt):
        """send message to iot hub"""

        # Create iothub client
        client = self.__iothub_client_init()

        # Check if no client is set
        if client is None:
            return False

        try:
            # Connect to iothub
            client.connect()

            print("Connection string:", self.__conf_iothub_connection_str)

            #formatted message to be sent to iot hub
            message = Message(msg_txt)
            print(message)
            #sends message to iot hub
            client.send_message(message)

        except CredentialError as ce:
            print("Credential error in connecting to client {0}".format(ce))
            logging.exception(ce)
            print("Sending alert failed...")
            result = False
        except ConnectionFailedError as cfe:
            print("Connection failed error in connecting to client {0}".format(cfe))
            logging.exception(cfe)
            print("Sending alert failed...")
            result = False
        except ConnectionDroppedError as cde:
            print("Connection dropped error in connecting to client {0}".format(cde))
            logging.exception(cde)
            print("Sending alert failed...")
            result = False
        except ClientError as clienterror:
            print("Client error in connecting to client {0}".format(clienterror))
            logging.exception(clienterror)
            print("Sending alert failed...")
            result = False
        except:
            print("Unknown error encountered.")
            print("Sending alert failed...")
            result = False
        else:
            print("Message successfully sent")
            result = True
        finally:
            client.disconnect()

        return result



