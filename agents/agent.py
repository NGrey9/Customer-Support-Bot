import os
import logging
from datetime import datetime
from dotenv import load_dotenv

from logs import LogFormatter
load_dotenv

LOGS_DIR = os.environ['LOGS_DIR']


class Agent:
    def __init__(self, logs_dir: str = LOGS_DIR, agent_name: str = None):
        self.agent_name = agent_name
        self.setup_logging(logs_dir=logs_dir)

    def setup_logging(self, logs_dir: str):
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)

        self.logger = logging.getLogger(self.agent_name)
        self.logger.setLevel(logging.DEBUG)

        fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

        detailed_log = os.path.join(
            logs_dir, f'{self.agent_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        file_handler = logging.FileHandler(detailed_log)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(LogFormatter(fmt))

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(LogFormatter(fmt))

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.query_logger = logging.getLogger('UserQueries')
        self.query_logger.setLevel(logging.INFO)
        query_log = os.path.join(
            logs_dir, f'{self.agent_name}_queries_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        query_handler = logging.FileHandler(query_log)
        query_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(message)s'))
        self.query_logger.addHandler(query_handler)
