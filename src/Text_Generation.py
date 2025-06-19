import os
import torch
import mlflow
import logging
from datetime import datetime
from typing import Dict, Tuple, Any

from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_scheduler,
    PreTrainedTokenizer,
    PreTrainedModel
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from accelerate import Accelerator
from tqdm import tqdm

from Common_Utils import setup_logger, track_performance, CustomException, load_config, upload_model_hub
from src.base_model import NLPBaseModel
import dagshub

os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

logger = setup_logger(filename="NLP_logger_test")
accelerator = Accelerator()
device = accelerator.device


class Text_Generation(NLPBaseModel):
    def __init__(self) -> None:
        """
        Initializes the base model class for text generation.
        """
        super().__init__()

    @track_performance
    def preprocess_function(self, examples: Dict[str, Any], tokenizer: PreTrainedTokenizer, cfg: Dict) -> Dict:
        """
        Preprocess the raw dataset into tokenized format with input and labels.

        Args:
            examples: A batch of dataset examples.
            tokenizer: HuggingFace tokenizer.
            cfg: Configuration dictionary.

        Returns:
            Dictionary of tokenized inputs and labels.
        """
        try:
            if tokenizer.pad_token is None:
                logger.warning("Tokenizer has no pad_token. Setting pad_token to eos_token.")
                tokenizer.pad_token = tokenizer.eos_token

            texts = examples[cfg["dataset"]["text_column"]]
            inputs = tokenizer(
                texts,
                max_length=cfg["tokenizer"]["max_length"],
                truncation=cfg["tokenizer"]["truncation"],
                padding=cfg["tokenizer"]["padding"],
            )
            inputs["labels"] = inputs["input_ids"].copy()
            return inputs
        except CustomException as e:
            logger.error(f"Error in preprocessing: {e}")
            raise

    @track_performance
    def get_dataloaders(
        self, tokenizer: PreTrainedTokenizer, cfg: Dict) -> Tuple[DataLoader, DataLoader]:
        """
        Loads dataset, tokenizes, and returns train/val dataloaders.

        Args:
            tokenizer: HuggingFace tokenizer.
            cfg: Configuration dictionary.

        Returns:
            Tuple containing training and validation DataLoaders.
        """
        try:
            dataset = load_dataset(cfg["dataset"]["name"], cfg["dataset"].get("config", None))
            train_ds = dataset["train"].select(range(cfg["dataset"]["train_subset_size"]))
            val_ds = dataset["validation"].select(range(cfg["dataset"]["val_subset_size"]))

            tokenized_train = train_ds.map(
                lambda x: self.preprocess_function(x, tokenizer, cfg),
                batched=True,
                remove_columns=train_ds.column_names
            )
            tokenized_val = val_ds.map(
                lambda x: self.preprocess_function(x, tokenizer, cfg),
                batched=True,
                remove_columns=val_ds.column_names
            )

            collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
            return (
                DataLoader(tokenized_train, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=collator),
                DataLoader(tokenized_val, batch_size=cfg["train"]["batch_size"], collate_fn=collator)
            )
        except CustomException as e:
            logger.error(f"Error creating dataloaders: {e}")
            raise

    @track_performance
    def get_model(self, cfg: Dict) -> PreTrainedModel:
        """
        Loads a pre-trained CausalLM model with QLoRA PEFT configuration.

        Args:
            cfg: Configuration dictionary.

        Returns:
            The prepared model.
        """
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["model"]["name"],
                load_in_4bit=cfg["model"]["load_in_4bit"],
                torch_dtype=getattr(torch, cfg["model"]["torch_dtype"]),
                device_map=cfg["model"]["device_map"]
            )
            model = prepare_model_for_kbit_training(model)
            peft_config = LoraConfig(
                task_type=getattr(TaskType, cfg["lora"]["task_type"]),
                r=cfg["lora"]["r"],
                lora_alpha=cfg["lora"]["alpha"],
                lora_dropout=cfg["lora"]["dropout"],
                bias=cfg["lora"]["bias"],
                target_modules=cfg["model"]["target_modules"]
            )
            return get_peft_model(model, peft_config)
        except CustomException as e:
            logger.error(f"Error loading model: {e}")
            raise

    @track_performance
    def train(self, model: PreTrainedModel, optimizer: torch.optim.Optimizer, scheduler: Any, train_loader: DataLoader, cfg: Dict) -> None:
        """
        Train the model.

        Args:
            model: HuggingFace model.
            optimizer: Optimizer.
            scheduler: Learning rate scheduler.
            train_loader: DataLoader for training data.
            cfg: Configuration dictionary.
        """
        try:
            for epoch in range(cfg["train"]["epochs"]):
                model.train()
                total_loss = 0
                loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']}")
                for batch in loop:
                    outputs = model(**batch)
                    loss = outputs.loss
                    accelerator.backward(loss)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    total_loss += loss.item()
                    loop.set_postfix(loss=loss.item())

                avg_loss = total_loss / len(train_loader)
                logger.info(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")
                mlflow.log_metric("train_loss", avg_loss, step=epoch)
        except CustomException as e:
            logger.error(f"Training failed: {e}")
            raise

    @track_performance
    def evaluate_model(self, model: PreTrainedModel, val_loader: DataLoader,tokenizer: PreTrainedTokenizer, cfg: Dict) -> Dict[str, float]:
        """
        Evaluate model using perplexity metric.

        Args:
            model: Trained model.
            val_loader: Validation DataLoader.
            tokenizer: Tokenizer used.
            cfg: Configuration dictionary.

        Returns:
            Dictionary containing perplexity score.
        """
        try:
            model.eval()
            perplexities = []
            for batch in tqdm(val_loader, desc="Evaluating"):
                with torch.no_grad():
                    outputs = model(**batch)
                    loss = outputs.loss
                    perplexities.append(torch.exp(loss).item())

            avg_perplexity = sum(perplexities) / len(perplexities)
            mlflow.log_metric("eval_perplexity", avg_perplexity)
            return {"perplexity": avg_perplexity}
        except CustomException as e:
            logger.error(f"Evaluation failed: {e}")
            raise

    @track_performance
    def save_model(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, cfg: Dict) -> None:
        """
        Save the trained model and tokenizer locally and log to MLflow.

        Args:
            model: Trained model.
            tokenizer: Tokenizer used.
            cfg: Configuration dictionary.
        """
        try:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(cfg["save"]["output_dir"], save_function=accelerator.save)
            tokenizer.save_pretrained(cfg["save"]["output_dir"])

            mlflow.pytorch.log_model(unwrapped_model, "model")
            for root, _, files in os.walk(cfg["save"]["output_dir"]):
                for file in files:
                    mlflow.log_artifact(os.path.join(root, file))

            if os.path.exists(cfg.get("config_path", "config.yaml")):
                mlflow.log_artifact(cfg["config_path"])
        
        except CustomException as e:
            logger.error(f"Saving model/tokenizer failed: {e}")
            raise


@track_performance
def execute_text_generation() -> None:
    """
    Run the end-to-end text generation workflow.
    """
    try:
        logger.info("Commencing Text generation workflow..")
        cfg = load_config("./Config_Yaml/config_text_generation.yaml")

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        mlflow.set_tracking_uri("https://github.com/anp102618/llm_nlp_tasks_mlops.git")
        mlflow.set_experiment(cfg["mlflow"]["experiment_name"] + "_" + timestamp)

        with mlflow.start_run(run_name=cfg["mlflow"]["run_name"]):
            for section, values in cfg.items():
                if isinstance(values, dict):
                    for key, val in values.items():
                        mlflow.log_param(f"{section}.{key}", val)

            tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
            tg = Text_Generation()
            model = tg.get_model(cfg)
            train_loader, val_loader = tg.get_dataloaders(tokenizer, cfg)

            train_loader.collate_fn.model = model
            val_loader.collate_fn.model = model

            optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["learning_rate"]))
            scheduler = get_scheduler(
                "linear",
                optimizer=optimizer,
                num_warmup_steps=cfg["train"]["warmup_steps"],
                num_training_steps=cfg["train"]["epochs"] * len(train_loader)
            )

            model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
                model, optimizer, train_loader, val_loader, scheduler
            )

            tg.train(model, optimizer, scheduler, train_loader, cfg)
            tg.evaluate_model(model, val_loader, tokenizer, cfg)
            tg.save_model(model, tokenizer, cfg)
            #upload_model_hub(model, tokenizer, cfg)

    except CustomException as e:
        logger.error(f"Execution failed: {e}")
        raise


if __name__ == "__main__":
    execute_text_generation()
