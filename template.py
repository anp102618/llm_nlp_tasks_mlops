import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

list_of_files = [
    "__init__.py",
    ".github/workflows/.gitkeep",
    ".github/workflows/deploy.yaml",

    f"Data/__init__.py",

    f"Common_Utils/__init__.py",
    

    f"src/__init__.py",
    f"src/Text_Classification.py",
    f"src/Text_Summarization.py",
    f"src/QA.py",
    f"src/Text_Generation.py",
    f"src/Machine_Translation.py",

    f"Config_Yaml/__init__.py",
    f"Config_Yaml/config_text_classification.yaml",
    f"Config_Yaml/config_text_summarization.yaml",
    f"Config_Yaml/config_qa.yaml",
    f"Config_Yaml/config_text_generation.yaml",
    f"Config_Yaml/config_machine_translation.yaml",

    "app.py",
    "streamlit_app.py",
    "requirements.txt",
    "setup.py",
    "main.py",
    "Dockerfile",
    ".dockerignore",
    ".gitignore"
 


]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory:{filedir} for the file {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath,'w') as f:
            pass
            logging.info(f"Creating empty file: {filepath}")

    else:
        logging.info(f"{filename} is already exists")