from dataclasses import dataclass
from pathlib import Path

from fau_rag_opt.constants.my_constants import CONFIG_FILE_PATH
from fau_rag_opt.helpers.utils import read_yaml


@dataclass(frozen = True)
class RetrieverConfig:
    transformer: Path

class ConfigurationManager:
    def __init__(self,
                config_filepath = CONFIG_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.retriever_config = RetrieverConfig(
            transformer = self.config["retriever_transformer"]["transformer"]
        )

    def get_retriever_config(self) -> RetrieverConfig:
        return self.retriever_config

# Global configuration instance - (Singleton design pattern)
config_manager = ConfigurationManager()
retriever_config = config_manager.get_retriever_config()