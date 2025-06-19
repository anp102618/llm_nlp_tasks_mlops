import os
import logging
import torch
import mlflow
from datetime import datetime
from typing import Dict, Tuple, List, Any
from torch.utils.data import DataLoader
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    get_scheduler,
    PreTrainedModel,
    PreTrainedTokenizerBase
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from accelerate import Accelerator
import evaluate
from tqdm import tqdm

from Common_Utils import setup_logger, track_performance, CustomException, load_config, upload_model_hub
from src.base_model import NLPBaseModel
import dagshub

os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

logger = setup_logger(filename="NLP_logger_test")
accelerator = Accelerator()
device = accelerator.device


class Text_Summarization(NLPBaseModel):
    def __init__(self):
        """Initialize Text Summarization task."""
        super().__init__()

    @track_performance
    def preprocess_function(self, examples: Dict[str, List[str]], tokenizer: PreTrainedTokenizerBase, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tokenizes input and target texts for summarization.

        Args:
            examples: Dataset batch with source and target text.
            tokenizer: HuggingFace tokenizer instance.
            cfg: Configuration dictionary.

        Returns:
            Dictionary with tokenized inputs and labels.
        """
        try:
            if tokenizer.pad_token is None:
                logger.warning("Tokenizer has no pad_token. Setting pad_token to eos_token.")
                tokenizer.pad_token = tokenizer.eos_token

            logger.info("Data preprocessing starting..")
            prefix = cfg["tokenizer"]["prefix"]
            inputs = tokenizer(
                [prefix + x for x in examples[cfg["dataset"]["text_column"]]],
                max_length=cfg["tokenizer"]["max_input_length"],
                truncation=cfg["tokenizer"]["truncation"],
                padding=cfg["tokenizer"]["padding"]
            )
            targets = tokenizer(
                examples[cfg["dataset"]["label_column"]],
                max_length=cfg["tokenizer"]["max_target_length"],
                truncation=cfg["tokenizer"]["truncation"],
                padding=cfg["tokenizer"]["padding"]
            )
            inputs["labels"] = targets["input_ids"]
            logger.info("Data preprocessing completed..")
            return inputs
        except CustomException as e:
            logger.error("Error in preprocessing", e)
            raise

    @track_performance
    def get_dataloaders(self, tokenizer: PreTrainedTokenizerBase, cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
        """
        Loads and tokenizes datasets, returns dataloaders.

        Args:
            tokenizer: HuggingFace tokenizer.
            cfg: Configuration dictionary.

        Returns:
            Tuple of training and validation dataloaders.
        """
        try:
            logger.info("Data loading and tokenization starting..")
            dataset: DatasetDict = load_dataset(cfg["dataset"]["name"], cfg["dataset"].get("config", None))
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

            collator = DataCollatorForSeq2Seq(tokenizer)
            train_loader = DataLoader(tokenized_train, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=collator)
            val_loader = DataLoader(tokenized_val, batch_size=cfg["train"]["batch_size"], collate_fn=collator)
            logger.info("Dataloaders created successfully.")
            return train_loader, val_loader
        
        except CustomException as e:
            logger.error("Error creating dataloaders", e)
            raise

    @track_performance
    def get_model(self, cfg: Dict[str, Any]) -> PreTrainedModel:
        """
        Loads and prepares model with LoRA.

        Args:
            cfg: Configuration dictionary.

        Returns:
            PEFT-wrapped Transformer model.
        """
        try:
            logger.info("Model import starting..")
            model = AutoModelForSeq2SeqLM.from_pretrained(
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
            model = get_peft_model(model, peft_config)
            logger.info("Model loaded and prepared with QLoRA.")
            return model
        
        except CustomException as e:
            logger.error("Error loading model", e)
            raise

    @track_performance
    def train(
        self,
        model: PreTrainedModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        train_loader: DataLoader,
        cfg: Dict[str, Any]) -> None:
        """
        Trains the model using the training dataloader.

        Args:
            model: PEFT Transformer model.
            optimizer: Optimizer.
            scheduler: Learning rate scheduler.
            train_loader: Training dataloader.
            cfg: Configuration dictionary.
        """
        try:
            logger.info("Model training starting..")
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
            logger.info("Model training completed successfully..")
        
        except CustomException as e:
            logger.error("Training failed", e)
            raise

    @track_performance
    def evaluate_model(
        self,
        model: PreTrainedModel,
        val_loader: DataLoader,
        tokenizer: PreTrainedTokenizerBase,
        cfg: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluates the model using ROUGE.

        Args:
            model: PEFT Transformer model.
            val_loader: Validation dataloader.
            tokenizer: Tokenizer.
            cfg: Configuration dictionary.

        Returns:
            Dictionary of evaluation metrics.
        """
        try:
            logger.info("Model evaluation starting..")
            rouge = evaluate.load(cfg["evaluation"]["metric"])
            model.eval()
            preds, labels = [], []

            for batch in tqdm(val_loader, desc="Evaluating"):
                with torch.no_grad():
                    generated_tokens = accelerator.unwrap_model(model).generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        max_length=cfg["evaluation"]["max_length"]
                    )
                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
                preds.extend(decoded_preds)
                labels.extend(decoded_labels)

            metrics = rouge.compute(predictions=preds, references=labels, use_stemmer=True)
            metrics = {k: round(v, 4) for k, v in metrics.items()}
            logger.info(f"Evaluation metrics: {metrics}")
            mlflow.log_metrics(metrics)
            return metrics
        
        except CustomException as e:
            logger.error("Evaluation failed", e)
            raise

    @track_performance
    def save_model(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, cfg: Dict[str, Any]) -> None:
        """
        Saves the model and tokenizer locally and logs them to MLflow.

        Args:
            model: PEFT Transformer model.
            tokenizer: Tokenizer.
            cfg: Configuration dictionary.
        """
        try:
            logger.info("Saving model and tokenizer locally...")
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(cfg["save"]["output_dir"], save_function=accelerator.save)
            tokenizer.save_pretrained(cfg["save"]["output_dir"])
            logger.info("Logging model to MLflow...")
            mlflow.pytorch.log_model(unwrapped_model, "model")

            for root, _, files in os.walk(cfg["save"]["output_dir"]):
                for file in files:
                    mlflow.log_artifact(os.path.join(root, file))

            config_path = cfg.get("config_path", "config.yaml")
            if os.path.exists(config_path):
                mlflow.log_artifact(config_path)
        except CustomException as e:
            logger.error("Saving model/tokenizer failed", e)
            raise


@track_performance
def execute_text_summarization() -> None:
    """
    Executes the end-to-end summarization pipeline.
    """
    try:
        logger.info("Commencing Text summarization workflow..")
        cfg = load_config("./Config_Yaml/config_text_summarization.yaml")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        mlflow.set_tracking_uri("https://github.com/anp102618/llm_nlp_tasks_mlops.git")
        mlflow.set_experiment(cfg["mlflow"]["experiment_name"] + "_" + timestamp)

        with mlflow.start_run(run_name=cfg["mlflow"]["run_name"]):
            for section, values in cfg.items():
                if isinstance(values, dict):
                    for key, val in values.items():
                        mlflow.log_param(f"{section}.{key}", val)
                else:
                    mlflow.log_param(section, values)

            tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
            ts = Text_Summarization()
            model = ts.get_model(cfg)
            train_loader, val_loader = ts.get_dataloaders(tokenizer, cfg)

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

            ts.train(model, optimizer, scheduler, train_loader, cfg)
            ts.evaluate_model(model, val_loader, tokenizer, cfg)
            ts.save_model(model, tokenizer, cfg)
            #upload_model_hub(model, tokenizer, cfg)
            logger.info("Text summarization task completed successfully.")
    
    except CustomException as e:
        logger.error(f"Execution failed : {e}")
        raise


if __name__ == "__main__":
    execute_text_summarization()
