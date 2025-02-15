import os
from datetime import datetime
import logging


from dotenv import load_dotenv

load_dotenv()

LOGS_DIR = os.environ['LOGS_DIR']


class LogFormatter(logging.Formatter):
    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[1;31m"
    reset = "\x1b[0m"

    def __init__(self, fmt):
        super().__init__()
        self.fmt = fmt
        self.FORMATS = {
            logging.DEBUG: self.grey + self.fmt + self.reset,
            logging.INFO: self.blue + self.fmt + self.reset,
            logging.WARNING: self.yellow + self.fmt + self.reset,
            logging.ERROR: self.red + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class SetupLogging:
    def __init__(self, agent, agent_name: str):
        self.agent = agent
        self.agent_name = agent_name

    def setup_logging(self, logs_dir: str = LOGS_DIR):
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)

        self.agent.logger = logging.getLogger(self.agent_name)
        self.agent.logger.setLevel(logging.DEBUG)

        fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

        detailed_log = os.path.join(
            logs_dir, f'{self.agent_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        file_handler = logging.FileHandler(detailed_log)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(LogFormatter(fmt))

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(LogFormatter(fmt))

        self.agent.logger.addHandler(file_handler)
        self.agent.logger.addHandler(console_handler)

        self.query_logger = logging.getLogger('UserQueries')
        self.query_logger.setLevel(logging.INFO)
        query_log = os.path.join(
            logs_dir, f'{self.agent_name}_queries_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        query_handler = logging.FileHandler(query_log)
        query_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(message)s'))
        self.query_logger.addHandler(query_handler)
