
from abc import ABC, abstractmethod

class NLPBaseModel(ABC):
    """
    Abstract base class defining the NLP model pipeline structure.
    """
    
    def __init__(self):
        """
        Base constructor for the NLP model interface.
        """
        pass
    

    @abstractmethod
    def preprocess_function(self, *args, **kwargs):
        """
        Preprocess the dataset entries using the tokenizer and config.
        """
        pass

    @abstractmethod
    def get_dataloaders(self, *args, **kwargs):
        """
        Create tokenized training and validation dataloaders.
        """
        pass

    @abstractmethod
    def get_model(self, *args, **kwargs):
        """
        Load and configure the model (with optional LoRA).
        """
        pass

    @abstractmethod
    def train(self, *args, **kwargs):
        """
        Train the model using specified training loop.
        """
        pass

    @abstractmethod
    def evaluate_model(self, *args, **kwargs):
        """
        Evaluate model predictions using given metric.
        """
        pass

    @abstractmethod
    def save_model(self, *args, **kwargs):
        """
        Save model and tokenizer locally and to MLflow.
        """
        pass