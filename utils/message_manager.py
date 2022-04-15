#!/usr/bin/env python3

import sys

sys.path.insert(0, '/*')

import math
import queue
import time
import threading
import send_message as sm
import config as cf
from os import path
from timeit import default_timer as timer
from enum import Enum, auto

class State(Enum):
    IDLE = auto()
    BUSY = auto()
    ERROR = auto()

# class MessageManager
# Description: Alert message manager
#               Forward alert message to send_message.py
# Paremeter: MessageSender
# Return value: None
class MessageManager(sm.MessageSender):

        __message_queue = []
        __queue_copy = []                               # Holds the message queue copy
        __message_manager = None                # Holds the instance of MessageManager

        __state = None                                  # Initial state is idle

        is_client_iothub_ok = False     # Holds flag if client connection for iothub is OK

        __user_config = None                    # Holds the user configuration instance

        MAX_SEND_RETRY = 5

        # function __init__()
        # Description: class constructor
        # Paremeter: None
        # Return value: None
        def __init__(self, config=None):
                """Class constructor"""

                # Check if config is none
                if config is None:
                        # Initialize config
                        config = cf.Config()    

                # Check message manager is none
                if self.__message_manager is None:

                        # Set user configuration instance
                        self.__user_config = config
                        # Make instance of MessageSender
                        sm.MessageSender.__init__(self, config)
                        # Make instance of Message Manager
                        self.__message_manager = super().__init__(config);

                        # Global variable for checking connection status during thread processing
                        global is_client_iothub_ok
                        is_client_iothub_ok = True    # Set initially to OK

                        self.__state = State.IDLE

                        self.MAX_SEND_RETRY = config.MAX_SEND_RETRY

        # function add()
        # Description: Function that message received are added to message stack pool. FIFO is followed.
        # Paremeter: self, msg_txt
        # Return value: None
        def add(self, msg_txt):
                """Add alert message to message queue"""

                # Add new alert message to message queue
                self.__message_queue.append(msg_txt)


        # function start()
        # Description: Function that start the message manager execution
        # Paremeter: None
        # Return value: None
        def start(self):
                """Start the message manager execution"""

                # Check if the state is idle
                if State.IDLE == self.__state:

                        # make state busy
                        self.__state = State.BUSY

                        # start process
                        self.__run()

                        # make state idle
                        self.__state = State.IDLE

                # Check if the state is busy
                elif State.BUSY == self.__state:

                        print("Can't start. Message manager is busy at this moment.")

                # Check if the state has error
                elif State.ERROR == self.__state:

                        print("Can't start. Client connections cannot established.")

        # function __run()
        # Description: Function to run process
        # Paremeter: None
        # Return value: None
        def __run(self):
                """Run message manager process"""

                # clear queue copy before use
                self.__queue_copy.clear()

                # Check if message queue is not empty
                while 0 < len(self.__message_queue):

                        # Move alert from message queue to queue copy
                        self.__queue_copy.append(self.__message_queue.pop(0))

                # Process every alert message to send them
                for message in self.__queue_copy:

                        # Execute each thread per alert
                        x = threading.Thread(target=self.send, args=(message,))
                        x.start()

        # function send()
        # Description: Function to call send_message function to send message
        # Paremeter: message
        # Return value: None
        def send(self, message):
                """Send alert message to iot hub"""

                # Connection status for alert thread processing
                global is_client_iothub_ok

                # get message details
                num = 0         # number of attempt to send message to IoT Hub

                print("===============================================================") 
                print("Send Message:", message)
                print("Destination: IoT Hub")

                # put retry limit when send alert failed
                while self.MAX_SEND_RETRY > num:

                        # add attempt
                        num = num + 1 
                        # send alert message for IoT hub
                        print(">>> Attempt:", num,", Sending...")   
                        if self.iothub_client_send_message(message):

                                # Set iothub connection status to ok
                                is_client_iothub_ok = True
                                break

                        else:

                                # Set iothub connection status to error
                                is_client_iothub_ok = False
                                print("Message sending failed!")

                                

        # function is_client_iothub_ok()
        # Description: Function that return the status of iothub client connection
        # Paremeter: None
        # Return value: is_client_iothub_ok
        def is_iothub_conn_ok(self):
                """Return true if the iothub client connection is OK. False if error occurred"""
                global is_client_iothub_ok
                return is_client_iothub_ok

        # function getUserConfig()
        # Description: Function that returns the user configuration instance
        # Paremeter: None
        # Return value: __user_config
        def getUserConfig(self):
                """Return the user configuraiton instance"""
                return self.__user_config






