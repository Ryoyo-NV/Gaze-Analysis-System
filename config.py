#!/usr/bin/env python3

import os
import configparser

# class Config
# Description: PFA user configuration
# Paremeter: None
# Return value: None
class Config:
	"""User configuration"""

	FILE_PATH_STR = "azure_config.ini" 	# Config file path

	# function __init__
	# Description: class constructor
	# Paremeter: None
	# Return value: None
	def __init__(self):
		"""Class constructor"""
		self.__init_config()

	# function __init_config
	# Description: Function to initialize config
	# Paremeter: None
	# Return value: None
	def __init_config(self):
		"""Configuration settings"""

		# User config option list
		HOST_NAME = "host_name"
		DEVICE_ID = "device_id"
		SHARED_ACCESS_KEY = "shared_access_key"

		# Set config file
		config = configparser.ConfigParser()

		# Check if config file is set
		if os.path.isfile(self.FILE_PATH_STR) is True:
			config.read(self.FILE_PATH_STR)
			config.optionxform = str 			# make options change to lower case
		else:
			# Since config file cannot be used, default values will be set
			print("[ERROR] Problem encountered upon opening the user config file.")
			print("[INFO] Please verify [{}]".format(self.FILE_PATH_STR))
			self.__set_defaults()
			return
		
		# read values from a section
		try:

			# Set iothub connection string from user config
			iot_hub_hostname = config.get('iot_hub_client_setting', HOST_NAME)
			iot_hub_device_id = config.get('iot_hub_client_setting', DEVICE_ID)
			iothub_shared_access_key = config.get('iot_hub_client_setting', SHARED_ACCESS_KEY)

		except:

			# Set default connection string
			self.__connection_string = "HostName={};DeviceId={};SharedAccessKey={}".format(\
				constant.HOST_NAME, constant.DEVICE_ID, constant.SHARED_ACCESS_KEY)

			print("[ERROR] Config set for CONNECTION_STRING is invalid! Default {} is used."\
				.format(self.__connection_string))

		else:

			self.__connection_string = "HostName={};DeviceId={};SharedAccessKey={}".format(\
				iot_hub_hostname, iot_hub_device_id, iothub_shared_access_key)

			print("CONNECTION_STRING config is set successfully!")

		finally:

			pass
 
		############################################################################################
		
	# function get_iot_hub_conn_string
	# Description: Function that returns the iothub connection string from user config
	# Paremeter: None
	# Return value: __connection_string
	def get_iot_hub_conn_string(self):
		"""Returns iothub connection string"""

		return self.__connection_string

	# function __set_defaults
	# Description: Function that set default for all config settings
	# Paremeter: None
	# Return value: None
	def __set_defaults(self):
		"""Default settings"""

		# Set default values since user config cannot use
		self.__connection_string = "HostName={};DeviceId={};SharedAccessKey={}".format(\
				constant.HOST_NAME, constant.DEVICE_ID, constant.SHARED_ACCESS_KEY)

		print("[INFO] Config default settings are used")
