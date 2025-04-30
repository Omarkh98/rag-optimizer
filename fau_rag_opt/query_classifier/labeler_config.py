# ------------------------------------------------------------------------------
# query_classifier/labeler_config.py - 
# ------------------------------------------------------------------------------
"""

"""

from dataclasses import dataclass

from ..constants.my_constants import CONFIG_FILE_PATH
from ..helpers.utils import read_yaml


@dataclass(frozen = True)
class LabelerConfig:
    base_url: str
    api_key: str
    model: str

class ConfigurationManager:
    def __init__(self,
                config_filepath = CONFIG_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.labeler_config = LabelerConfig(
            base_url = self.config["fau_llm"]["base_url"],
            api_key = self.config["fau_llm"]["api_key"],
            model = self.config["fau_llm"]["model"]
        )

    def get_labeler_config(self) -> LabelerConfig:
        return self.labeler_config

# Global configuration instance - (Singleton design pattern)
config_manager = ConfigurationManager()
labeler_config = config_manager.get_labeler_config()