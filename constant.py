
####################################################################################################
##################################### User Configurtion ############################################
####################################################################################################
# How to edit:
# 1. Below variables are grouped via header --> [<section name>]. e.g [display_setting]
#	 Please do not change any group header.
# 2. Every group is separated by the line of '#' for readability.
# 3. Comment above each variable indicates the purpose of it.
# 4. Please do not change any variable name. Variables are named all capital letters.
# 5. To edit the value, update the value after '=' operator. The value should be input 1 space after
#    '=' operator. Do not put any trailing after value like comment, spaces, etc.
# 6. Save the file when your done.
####################################################################################################
####################################################################################################	
####################################################################################################
# IOT Hub Client Setting
# - This group will be used for IOT Hub client connection	
#[iot_hub_client_setting]

# IOT Hub Client connection string
# - This connection string will be provided after an iot device such as jetson nano is registered in
#	Azure Cloud account.
HOST_NAME = "signage-ad.azure-devices.net"
DEVICE_ID = "signage-nx2"
SHARED_ACCESS_KEY = "xxxxxxx"
#HOST_NAME = "<iot hub hostname>"
#DEVICE_ID = "<iot device id>"
#SHARED_ACCESS_KEY = "<iot device shared access key>"


###############################################################
# Components
###############################################################

# Connection status
CONN_STAT_FONT_COLOR = (225, 225, 225) 	# Connection status font color
CONN_STAT_BG_COLOR = (0, 0, 0)			# Connection status background color
CONN_STAT_IOT_PT = (5, 10)			# Connection status for iothub X and Y
CONN_STAT_BG_PT1 = (00, 00)			# Connection status background X and Y
CONN_STAT_BG_PT2 = (190, 28)		# Connection status background X + Width and Y + Height
CONN_STAT_BG_PT3 = (190, 18)		# Connection status background X + Width and Y + Height
CONN_STAT_FONT_SIZE = 0.35			# Connection status font size



