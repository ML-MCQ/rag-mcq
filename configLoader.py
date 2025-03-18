import os
import yaml


def load_yaml_file(file_path):
    """
    Load a YAML file and return its contents as a Python dictionary.
    The file_path is resolved relative to the location of this script.
    """
    # Determine the absolute path relative to this file's directory
    full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)

    with open(full_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config():
    """ "
    Load configuration files from the 'conf' directory.
    Returns a dictionary with keys for each configuration section.
    """
    config = {}
    config["vectorStore"] = load_yaml_file("conf/vector_store.yml")
    config["questionGeneration"] = load_yaml_file("conf/question_generation.yml")
    config["pdfProcessing"] = load_yaml_file("conf/pdf.yml")
    config["embeddingModel"] = load_yaml_file("conf/embedding.yml")

    return config
