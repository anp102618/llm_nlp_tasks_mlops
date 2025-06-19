import os
import logging
import torch
import mlflow
from datetime import datetime
from typing import Tuple, Dict, Any
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_scheduler,
    PreTrainedTokenizer,
    PreTrainedModel,
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


class Text_Classification(NLPBaseModel):
    """
    A class for handling text classification using Transformers and PEFT (QLoRA).
    """

    def __init__(self) -> None:
        super().__init__()

    @track_performance
    def preprocess_function(self, examples: Dict[str, Any], tokenizer: PreTrainedTokenizer, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tokenizes the dataset for classification.

        Args:
            examples (dict): Dataset examples.
            tokenizer (PreTrainedTokenizer): Tokenizer object.
            cfg (dict): Configuration dictionary.

        Returns:
            dict: Tokenized inputs.
        """
        try:
            if tokenizer.pad_token is None:
                logger.warning("Tokenizer has no pad_token. Setting pad_token to eos_token.")
                tokenizer.pad_token = tokenizer.eos_token

            texts = examples[cfg["dataset"]["text_column"]]
            labels = examples[cfg["dataset"]["label_column"]]

            inputs = tokenizer(
                texts,
                max_length=cfg["tokenizer"]["max_length"],
                truncation=cfg["tokenizer"]["truncation"],
                padding=cfg["tokenizer"]["padding"],
            )
            inputs["labels"] = labels
            return inputs
        except CustomException as e:
            logger.error(f"Preprocessing error: {e}")
            raise

    @track_performance
    def get_dataloaders(self, tokenizer: PreTrainedTokenizer, cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
        """
        Loads, tokenizes and returns train/validation dataloaders.

        Args:
            tokenizer (PreTrainedTokenizer): Tokenizer object.
            cfg (dict): Configuration dictionary.

        Returns:
            Tuple[DataLoader, DataLoader]: Train and validation dataloaders.
        """
        try:
            dataset: DatasetDict = load_dataset(cfg["dataset"]["name"], cfg["dataset"].get("config", None))
            train_ds = dataset["train"].select(range(cfg["dataset"]["train_subset_size"]))
            val_ds = dataset["validation"].select(range(cfg["dataset"]["val_subset_size"]))

            tokenized_train = train_ds.map(
                lambda x: self.preprocess_function(x, tokenizer, cfg),
                batched=True,
                remove_columns=train_ds.column_names,
            )
            tokenized_val = val_ds.map(
                lambda x: self.preprocess_function(x, tokenizer, cfg),
                batched=True,
                remove_columns=val_ds.column_names,
            )

            collator = DataCollatorWithPadding(tokenizer)
            train_loader = DataLoader(tokenized_train, batch_size=cfg["train"]["batch_size"], collate_fn=collator, shuffle=True)
            val_loader = DataLoader(tokenized_val, batch_size=cfg["train"]["batch_size"], collate_fn=collator)

            return train_loader, val_loader
        except CustomException as e:
            logger.error(f"Error creating dataloaders: {e}")
            raise

    @track_performance
    def get_model(self, cfg: Dict[str, Any]) -> PreTrainedModel:
        """
        Loads the transformer model with optional LoRA.

        Args:
            cfg (dict): Model configuration dictionary.

        Returns:
            PreTrainedModel: Prepared model instance.
        """
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                cfg["model"]["name"],
                load_in_4bit=cfg["model"]["load_in_4bit"],
                torch_dtype=getattr(torch, cfg["model"]["torch_dtype"]),
                device_map=cfg["model"]["device_map"],
            )
            model = prepare_model_for_kbit_training(model)

            peft_config = LoraConfig(
                task_type=getattr(TaskType, cfg["lora"]["task_type"]),
                r=cfg["lora"]["r"],
                lora_alpha=cfg["lora"]["alpha"],
                lora_dropout=cfg["lora"]["dropout"],
                bias=cfg["lora"]["bias"],
                target_modules=cfg["model"]["target_modules"],
            )

            model = get_peft_model(model, peft_config)
            return model
        except CustomException as e:
            logger.error(f"Error loading model: {e}")
            raise

    @track_performance
    def train(self, model: PreTrainedModel, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler, 
              train_loader: DataLoader, cfg: Dict[str, Any]) -> None:
        """
        Trains the model on the provided dataloader.

        Args:
            model (PreTrainedModel): The model.
            optimizer (Optimizer): The optimizer.
            scheduler (_LRScheduler): The learning rate scheduler.
            train_loader (DataLoader): Training data.
            cfg (dict): Configuration dictionary.
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
                mlflow.log_metric("train_loss", avg_loss, step=epoch)
        except CustomException as e:
            logger.error(f"Training error: {e}")
            raise

    @track_performance
    def evaluate_model(self, model: PreTrainedModel, val_loader: DataLoader, cfg: Dict[str, Any]) -> float:
        """
        Evaluates the model and logs accuracy.

        Args:
            model (PreTrainedModel): The trained model.
            val_loader (DataLoader): Validation data.
            cfg (dict): Configuration.

        Returns:
            float: Accuracy score.
        """
        try:
            model.eval()
            correct, total = 0, 0

            for batch in tqdm(val_loader, desc="Evaluating"):
                with torch.no_grad():
                    outputs = accelerator.unwrap_model(model)(**batch)
                    preds = torch.argmax(outputs.logits, dim=-1)
                    labels = batch["labels"]
                    total += labels.size(0)
                    correct += (preds == labels).sum().item()

            accuracy = correct / total
            mlflow.log_metric("val_accuracy", accuracy)
            return accuracy
        except CustomException as e:
            logger.error(f"Evaluation error: {e}")
            raise

    @track_performance
    def save_model(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, cfg: Dict[str, Any]) -> None:
        """
        Saves the trained model and tokenizer to disk and logs them to MLflow.

        Args:
            model (PreTrainedModel): The model.
            tokenizer (PreTrainedTokenizer): The tokenizer.
            cfg (dict): Save configuration.
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

            config_path = cfg.get("config_path", "config.yaml")
            if os.path.exists(config_path):
                mlflow.log_artifact(config_path)
        except CustomException as e:
            logger.error(f"Model saving error: {e}")
            raise


@track_performance
def execute_text_classification() -> None:
    """
    Executes the full text classification pipeline:
    - Loads config
    - Loads and tokenizes data
    - Initializes model with QLoRA
    - Trains, evaluates, and saves artifacts
    """
    try:
        dagshub_username = os.getenv("DAGSHUB_USERNAME")
        dagshub_token = os.getenv("DAGSHUB_TOKEN")

        if dagshub_username and dagshub_token:
            os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_username
            os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        else:
            raise ValueError("DAGSHUB_USERNAME and/or DAGSHUB_TOKEN environment variables are not set.")
        logger.info("Commencing Text classification workflow...")
        cfg = load_config("./Config_Yaml/config_text_classification.yaml")

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        current_model = f"{cfg['mlflow']['experiment_name']}_{timestamp}"

        mlflow.set_tracking_uri("https://github.com/anp102618/llm_nlp_tasks_mlops.git")
        experiment_name = cfg["mlflow"]["experiment_name"]
        run_name = f"{current_model}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            logger.info(f"Started MLflow run: {run_id}")
            for section, values in cfg.items():
                if isinstance(values, dict):
                    for key, val in values.items():
                        mlflow.log_param(f"{section}.{key}", val)

            tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
            tc = Text_Classification()
            model = tc.get_model(cfg)
            train_loader, val_loader = tc.get_dataloaders(tokenizer, cfg)

            optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["learning_rate"]))
            scheduler = get_scheduler(
                "linear",
                optimizer=optimizer,
                num_warmup_steps=cfg["train"]["warmup_steps"],
                num_training_steps=cfg["train"]["epochs"] * len(train_loader),
            )

            model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
                model, optimizer, train_loader, val_loader, scheduler
            )

            tc.train(model, optimizer, scheduler, train_loader, cfg)
            tc.evaluate_model(model, val_loader, cfg)
            tc.save_model(model, tokenizer, cfg)
            #upload_model_hub(model, tokenizer, cfg)

            logger.info("Text classification task completed successfully.")
    except CustomException as e:
        logger.error(f"Execution failed: {e}")
        raise


if __name__ == "__main__":
    execute_text_classification()
