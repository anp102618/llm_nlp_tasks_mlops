import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import pandas as pd
import numpy as np
import os 
from src.Machine_Translation import execute_machine_translation
from src.QA import execute_qa
from src.Text_Classification import execute_text_classification
from src.Text_Generation import execute_text_generation
from src.Text_Summarization import execute_text_summarization
from src.implementation import execute_implementation
from Common_Utils import setup_logger, track_performance , CustomException, load_config

logger = setup_logger(filename="NLP_logger_test")

@track_performance
def pipeline():
    try: 
        logger.info(f"Starting training of NLP tasks..")
        #execute_machine_translation()
       # execute_qa()
        execute_text_classification()
       # execute_text_generation()
        #execute_text_summarization()
        execute_implementation()
        logger.info(f"training of NLP tasks completed succesfully..")

    except CustomException as e:
            logger.error(f"Error in preprocessing: {e}")
            raise


if __name__ == "__main__":
    pipeline()