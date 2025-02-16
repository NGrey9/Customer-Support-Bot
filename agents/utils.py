import os
import json
from dotenv import load_dotenv

load_dotenv()

# Product Description
COMMANDS_CONFIG_PATH = os.environ['COMMANDS_CONFIG_PATH']


def load_commands(agent_task: str) -> list:
    with open(COMMANDS_CONFIG_PATH, 'r') as f:
        commands = json.load(f)[agent_task]
    return commands
