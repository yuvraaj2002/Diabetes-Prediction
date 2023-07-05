import logging
import os
from datetime import datetime

# Name of the log file
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Path of directory where we will store logs
logs_path = os.path.join(os.getcwd(), "Logs", LOG_FILE)

# Creating Logs directory
os.makedirs(logs_path, exist_ok=True)

# Path of the log file
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)


logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# if __name__ == "__main__":
#     logging.info("Logging is working fine")