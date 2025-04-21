import json
import sys
import logging
from typing import List, Dict
from fau_rag_opt.helpers.exception import CustomException

class MetadataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Dict]:
        """
        Load metadata file into a list of dictionaries
        """
        metadata = []
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    metadata.append(json.loads(line))
            return metadata
        except Exception as e:
            logging.info(f"Failed to load metadata from {self.file_path}: {e}")
            raise CustomException(e, sys)