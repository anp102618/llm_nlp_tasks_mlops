from abc import ABC, abstractmethod
from typing import Any, Dict
from transformers import pipeline, Pipeline
import yaml
import gc
from Common_Utils import setup_logger, track_performance, CustomException, load_config

logger = setup_logger(filename="NLP_logger_test")

# ------------------- Utility -------------------

def load_config(path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        path (str): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ------------------- Strategy Pattern Base -------------------

class NLPTaskStrategy(ABC):
    """
    Abstract base class for NLP task strategies.
    All task-specific strategies must inherit from this class and implement the `run` method.
    """
    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """
        Executes the strategy logic.
        """
        pass

# ------------------- Strategy Implementations -------------------

class TextGenerationStrategy(NLPTaskStrategy):
    """
    Strategy for performing text generation.
    """
    def __init__(self, pipe: Pipeline):
        self.pipeline = pipe

    def run(self, prompt: str, **kwargs: Any) -> Any:
        """
        Generate text given a prompt.

        Args:
            prompt (str): Input prompt for text generation.
            **kwargs: Additional generation parameters.

        Returns:
            Any: Generated text.
        """
        return self.pipeline(prompt, **kwargs)


class TextClassificationStrategy(NLPTaskStrategy):
    """
    Strategy for performing text classification.
    """
    def __init__(self, pipe: Pipeline):
        self.pipeline = pipe

    def run(self, text: str) -> Any:
        """
        Classify the input text.

        Args:
            text (str): Input text.

        Returns:
            Any: Classification label and score.
        """
        return self.pipeline(text)


class TextSummarizationStrategy(NLPTaskStrategy):
    """
    Strategy for summarizing text.
    """
    def __init__(self, pipe: Pipeline):
        self.pipeline = pipe

    def run(self, text: str, **kwargs: Any) -> Any:
        """
        Summarize the input text.

        Args:
            text (str): Input document.
            **kwargs: Additional generation parameters.

        Returns:
            Any: Generated summary.
        """
        return self.pipeline(text, **kwargs)


class QuestionAnsweringStrategy(NLPTaskStrategy):
    """
    Strategy for answering questions using a context passage.
    """
    def __init__(self, pipe: Pipeline):
        self.pipeline = pipe

    def run(self, question: str, context: str) -> Any:
        """
        Answer a question using the provided context.

        Args:
            question (str): Question string.
            context (str): Contextual information.

        Returns:
            Any: Predicted answer span.
        """
        return self.pipeline(question=question, context=context)


class MachineTranslationStrategy(NLPTaskStrategy):
    """
    Strategy for translating text from English to Spanish.
    """
    def __init__(self, pipe: Pipeline):
        self.pipeline = pipe

    def run(self, text: str, **kwargs: Any) -> Any:
        """
        Translate the input English text into Spanish.

        Args:
            text (str): English sentence.
            **kwargs: Additional translation parameters.

        Returns:
            Any: Translated Spanish sentence.
        """
        return self.pipeline(text, **kwargs)

# ------------------- Context Class -------------------

class NLPContext:
    """
    Context class to switch and execute various NLP task strategies.
    """
    def __init__(self) -> None:
        self.strategy: NLPTaskStrategy | None = None

    def set_strategy(self, strategy: NLPTaskStrategy) -> None:
        """
        Set the NLP task strategy.

        Args:
            strategy (NLPTaskStrategy): Instance of a strategy.
        """
        self.strategy = strategy

    def execute(self, **kwargs: Any) -> Any:
        """
        Execute the selected strategy.

        Returns:
            Any: Result from the strategy.
        """
        if not self.strategy:
            raise ValueError("Strategy not set")
        return self.strategy.run(**kwargs)

# ------------------- Task Pipeline Executors -------------------
@track_performance
def execute_text_generation_pipeline(path: str) -> None:
    """
    Load and execute the text generation pipeline using configuration.

    Args:
        path (str): Path to YAML config file.
    """
    try:
        logger.info("Executing text generation pipeline...")
        cfg = load_config(path)
        generation_pipe = pipeline("text-generation", model=cfg["save"]["output_dir"], tokenizer=cfg["save"]["output_dir"])
        context = NLPContext()
        context.set_strategy(TextGenerationStrategy(generation_pipe))
        result = context.execute(
            prompt=cfg["inference"]["prompt"],
            max_length=cfg["inference"]["max_length"],
            do_sample=cfg["inference"]["do_sample"],
            num_return_sequences=cfg["inference"]["num_return_sequences"]
        )
        logger.info(f"Text Generation Result: {result}")
        print("Text Generation:", result)
    
    except CustomException as e:
        logger.error(f"Text generation pipeline failed: {e}")
        raise
    
    finally:
        del context
        gc.collect()

@track_performance
def execute_text_classification_pipeline(path: str) -> None:
    """
    Load and execute the text classification pipeline.

    Args:
        path (str): Path to YAML config file.
    """
    try:
        logger.info("Executing text classification pipeline...")
        cfg = load_config(path)
        classification_pipe = pipeline("text-classification", model=cfg["save"]["output_dir"], tokenizer=cfg["save"]["output_dir"])
        context = NLPContext()
        context.set_strategy(TextClassificationStrategy(classification_pipe))
        result = context.execute(text=cfg["inference"]["example_text"])
        logger.info(f"Text Classification Result: {result}")
        print("Text Classification:", result)
    
    except CustomException as e:
        logger.error(f"Text classification pipeline failed: {e}")
        raise

    finally:
        del context
        gc.collect()

@track_performance
def execute_text_summarization_pipeline(path: str) -> None:
    """
    Load and execute the text summarization pipeline.

    Args:
        path (str): Path to YAML config file.
    """
    try:
        logger.info("Executing text summarization pipeline...")
        cfg = load_config(path)
        summarization_pipe = pipeline("summarization", model=cfg["save"]["output_dir"], tokenizer=cfg["save"]["output_dir"])
        context = NLPContext()
        context.set_strategy(TextSummarizationStrategy(summarization_pipe))
        result = context.execute(
            text=cfg["inference"]["example_text"],
            max_length=cfg["inference"]["max_length"],
            min_length=cfg["inference"]["min_length"],
            do_sample=cfg["inference"]["do_sample"]
        )
        logger.info(f"Summarization Result: {result}")
        print("Summarization:", result)
    
    except CustomException as e:
        logger.error(f"Text summarization pipeline failed: {e}")
        raise

    finally:
        del context
        gc.collect()

@track_performance
def execute_question_answering_pipeline(path: str) -> None:
    """
    Load and execute the question answering pipeline.

    Args:
        path (str): Path to YAML config file.
    """
    try:
        logger.info("Executing question answering pipeline...")
        cfg = load_config(path)
        qa_pipe = pipeline("question-answering", model=cfg["save"]["output_dir"], tokenizer=cfg["save"]["output_dir"])
        context = NLPContext()
        context.set_strategy(QuestionAnsweringStrategy(qa_pipe))
        result = context.execute(
            question=cfg["inference"]["example_question"],
            context=cfg["inference"]["example_context"]
        )
        logger.info(f"QA Result: {result}")
        print("Answer:", result)

    except CustomException as e:
        logger.error(f"Question answering pipeline failed: {e}")
        raise

    finally:
        del context
        gc.collect()

@track_performance
def execute_machine_translation_pipeline(path: str) -> None:
    """
    Load and execute the machine translation pipeline.

    Args:
        path (str): Path to YAML config file.
    """
    try:
        logger.info("Executing machine translation pipeline...")
        cfg = load_config(path)
        translation_pipe = pipeline("translation_en_to_es", model=cfg["save"]["output_dir"], tokenizer=cfg["save"]["output_dir"])
        context = NLPContext()
        context.set_strategy(MachineTranslationStrategy(translation_pipe))
        result = context.execute(text=cfg["inference"]["example_text"])
        logger.info(f"Translation Result: {result}")
        print("Translation:", result)

    except CustomException as e:
        logger.error(f"Machine translation pipeline failed: {e}")
        raise

    finally:
        del context
        gc.collect()

# ------------------- Orchestration -------------------
@track_performance
def execute_implementation() -> None:
    """
    Run all NLP pipelines sequentially using their respective configurations.
    """
    logger.info("Starting full pipeline execution...")
    try:
        execute_text_generation_pipeline("Config_Yaml/config_text_generation.yaml")
        execute_text_classification_pipeline("Config_Yaml/config_text_classification.yaml")
        execute_text_summarization_pipeline("Config_Yaml/config_text_summarization.yaml")
        execute_question_answering_pipeline("Config_Yaml/config_qa.yaml")
        execute_machine_translation_pipeline("Config_Yaml/config_machine_translation.yaml")
        logger.info("All pipelines executed successfully.")
    
    except CustomException as e:
        logger.critical(f"Pipeline execution failed: {e}")
        raise

if __name__ == "__main__":
    execute_implementation()