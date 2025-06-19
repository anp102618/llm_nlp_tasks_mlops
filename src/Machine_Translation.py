from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Tuple, Optional, List
import os
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    DataCollatorForSeq2Seq,
    get_scheduler,
)
from accelerate import Accelerator
from datasets import DatasetDict, load_dataset
import evaluate
from tqdm import tqdm
import mlflow
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig, TaskType
from Common_Utils import setup_logger, track_performance, load_config, CustomException, upload_model_hub
from src.base_model import NLPBaseModel
import dagshub

os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")


logger = setup_logger(filename="NLP_logger_test")

accelerator = Accelerator()
device = accelerator.device

class Machine_Translation(NLPBaseModel):

    def __init__(self) -> None:
        """
        Initializes the Question_Answering model interface.
        """
        super().__init__()


    @track_performance
    def preprocess_function(self, examples: Dict[str, Any], tokenizer: PreTrainedTokenizer, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tokenizes and formats the source and target translations.

        Args:
            examples (dict): A batch of translation samples.
            tokenizer (AutoTokenizer): Tokenizer instance.
            cfg (dict): Configuration dictionary.

        Returns:
            dict: Tokenized inputs with labels for model training.
        """
        try:
            if tokenizer.pad_token is None:
                logger.warning("Tokenizer has no pad_token. Setting pad_token to eos_token.")
                tokenizer.pad_token = tokenizer.eos_token

            prefix = cfg["tokenizer"]["prefix"]
            src_lang = cfg["dataset"]["source_lang"]
            tgt_lang = cfg["dataset"]["target_lang"]

            inputs = tokenizer(
                [prefix + x[src_lang] for x in examples["translation"]],
                max_length=cfg["tokenizer"]["max_input_length"],
                truncation=cfg["tokenizer"]["truncation"],
                padding=cfg["tokenizer"]["padding"],
            )
            targets = tokenizer(
                [x[tgt_lang] for x in examples["translation"]],
                max_length=cfg["tokenizer"]["max_target_length"],
                truncation=cfg["tokenizer"]["truncation"],
                padding=cfg["tokenizer"]["padding"],
            )

            labels = [
                [(token if token != tokenizer.pad_token_id else -100) for token in label]
                for label in targets["input_ids"]
            ]
            inputs["labels"] = labels
            return inputs
        except CustomException as e:
            logger.error(f"Error in preprocessing: {e}")
            raise

    @track_performance
    def get_dataloaders(self, tokenizer: PreTrainedTokenizer, cfg: Dict[str, Any], model: Optional[PreTrainedModel] = None) -> Tuple[DataLoader, DataLoader]:
        """
        Loads dataset, applies preprocessing, and returns DataLoaders.

        Args:
            tokenizer (AutoTokenizer): Tokenizer instance.
            cfg (dict): Configuration dictionary.
            model (PreTrainedModel, optional): Model used by the DataCollator.

        Returns:
            tuple: Training and validation DataLoader objects.
        """
        try:
            logger.info("Loading dataset and preparing dataloaders...")
            dataset: DatasetDict = load_dataset(cfg["dataset"]["name"], cfg["dataset"]["config"])
            splits = dataset.keys()

            train_ds = dataset["train"].select(range(cfg["dataset"]["train_subset_size"]))

            if "validation" in splits:
                val_ds = dataset["validation"].select(range(cfg["dataset"]["val_subset_size"]))
            elif "test" in splits:
                val_ds = dataset["test"].select(range(cfg["dataset"]["val_subset_size"]))
            else:
                logger.warning("No validation/test split found; using part of training data as validation.")
                val_start = cfg["dataset"]["train_subset_size"]
                val_end = val_start + cfg["dataset"]["val_subset_size"]
                val_ds = dataset["train"].select(range(val_start, val_end))

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

            collator = DataCollatorForSeq2Seq(tokenizer, model=model)

            train_loader = DataLoader(
                tokenized_train, batch_size=cfg["train"]["batch_size"], collate_fn=collator, shuffle=True
            )
            val_loader = DataLoader(
                tokenized_val, batch_size=cfg["train"]["batch_size"], collate_fn=collator
            )

            logger.info("Dataloaders created successfully.")
            return train_loader, val_loader
        except CustomException as e:
            logger.error(f"Error creating dataloaders: {e}")
            raise

    @track_performance
    def get_model(self, cfg: Dict[str, Any]) -> PreTrainedModel:
        """
        Loads a Seq2Seq model with optional LoRA configuration.

        Args:
            cfg (dict): Model configuration dictionary.

        Returns:
            PreTrainedModel: Configured transformer model.
        """
        try:
            logger.info("Loading pretrained model...")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                cfg["model"]["name"],
                load_in_4bit=cfg["model"]["load_in_4bit"],
                torch_dtype=getattr(torch, cfg["model"]["torch_dtype"]),
                device_map=cfg["model"]["device_map"],
            )
            model = prepare_model_for_kbit_training(model)

            target_modules = cfg["model"].get("target_modules", None)
            if not target_modules or target_modules in [[], "", "null", None]:
                logger.warning("No target_modules specified. LoRA auto-detection enabled.")
                target_modules = None

            if target_modules:
                peft_config = LoraConfig(
                    task_type=TaskType.SEQ_2_SEQ_LM,
                    r=cfg["lora"]["r"],
                    lora_alpha=cfg["lora"]["alpha"],
                    lora_dropout=cfg["lora"]["dropout"],
                    bias=cfg["lora"]["bias"],
                    target_modules=target_modules,
                )
                model = get_peft_model(model, peft_config)
                logger.info("LoRA enabled.")
            else:
                logger.info("LoRA skipped; using base model only.")

            return model
        except CustomException as e:
            logger.error(f"Model loading error: {e}")
            raise

    @track_performance
    def train(self, model: PreTrainedModel, optimizer: Optimizer, scheduler: Any, train_loader: DataLoader,cfg: Dict[str, Any]) -> None:
        """
        Training loop for fine-tuning the transformer model.

        Args:
            model (PreTrainedModel): The transformer model.
            optimizer (torch.optim.Optimizer): Optimizer.
            scheduler (torch.optim.lr_scheduler): Scheduler.
            train_loader (DataLoader): DataLoader for training data.
            cfg (dict): Training configuration.

        Returns:
            None
        """
        try:
            logger.info("Starting training loop...")
            epochs = cfg["train"]["epochs"]

            for epoch in range(epochs):
                model.train()
                total_loss = 0.0
                loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

                for batch in loop:
                    outputs = model(**batch)
                    loss = outputs.loss

                    assert loss.requires_grad, "Loss does not require grad! Check model and inputs."

                    accelerator.backward(loss)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    total_loss += loss.item()
                    loop.set_postfix(loss=loss.item())

                avg_loss = total_loss / len(train_loader)
                logger.info(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
                mlflow.log_metric("train_loss", avg_loss, step=epoch)

            logger.info("Training completed successfully.")
        except CustomException as e:
            logger.error(f"Training error: {e}")
            raise

    @track_performance
    def evaluate_model(self, model: PreTrainedModel, val_loader: DataLoader, tokenizer: PreTrainedTokenizer, cfg: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluates the model on the validation set.

        Args:
            model (PreTrainedModel): The trained model.
            val_loader (DataLoader): Validation DataLoader.
            tokenizer (AutoTokenizer): Tokenizer.
            cfg (dict): Evaluation configuration.

        Returns:
            dict: Evaluation metrics (e.g., BLEU, ROUGE).
        """
        try:
            logger.info("Starting evaluation...")
            metric_name = cfg["evaluation"]["metric"]
            metric = evaluate.load(metric_name)

            model.eval()
            preds, labels = [], []

            for batch in tqdm(val_loader, desc="Evaluating"):
                with torch.no_grad():
                    generated_tokens = accelerator.unwrap_model(model).generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        max_length=cfg["evaluation"]["max_length"],
                    )
                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                labels_clean = [
                    [token if token != -100 else tokenizer.pad_token_id for token in label]
                    for label in batch["labels"]
                ]
                decoded_labels = tokenizer.batch_decode(labels_clean, skip_special_tokens=True)

                preds.extend(decoded_preds)
                labels.extend(decoded_labels)

            if metric_name == "rouge":
                metrics = metric.compute(predictions=preds, references=labels, use_stemmer=True)
            elif metric_name == "bleu":
                metrics = metric.compute(predictions=preds, references=[[ref] for ref in labels])
            else:
                metrics = metric.compute(predictions=preds, references=labels)

            metrics = {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}
            logger.info(f"Evaluation metrics: {metrics}")
            scalar_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
            mlflow.log_metrics(scalar_metrics)

            return metrics
        except CustomException as e:
            logger.error(f"Evaluation error: {e}")
            raise

    @track_performance
    def save_model(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, cfg: Dict[str, Any]) -> None:
        """
        Saves the model and tokenizer to disk and logs them to MLflow.

        Args:
            model (PreTrainedModel): Trained model.
            tokenizer (AutoTokenizer): Tokenizer.
            cfg (dict): Configuration with save path.

        Returns:
            None
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

            logger.info("Model and tokenizer saved and logged successfully.")
        except CustomException as e:
            logger.error(f"Saving error: {e}")
            raise


@track_performance
def execute_machine_translation() -> None:
    """
    Orchestrates the complete machine translation pipeline:
    - Loads config and tokenizer
    - Loads and trains model
    - Evaluates and saves artifacts

    Returns:
        None
    """
    try:
        logger.info("Commencing Machine Translation workflow..")
        cfg: Dict[str, Any] = load_config("./Config_Yaml/config_machine_translation.yaml")

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        current_model = f"{cfg['mlflow']['experiment_name']}_{timestamp}"

        mlflow.set_tracking_uri("https://github.com/anp102618/llm_nlp_tasks_mlops.git")
        mlflow.set_experiment(current_model)

        with mlflow.start_run(run_name=cfg["mlflow"]["run_name"]):
            for section, values in cfg.items():
                if isinstance(values, dict):
                    for key, val in values.items():
                        mlflow.log_param(f"{section}.{key}", val)
                else:
                    mlflow.log_param(section, values)

            tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
            mt = Machine_Translation()
            model = mt.get_model(cfg)
            train_loader, val_loader = mt.get_dataloaders(tokenizer, cfg, model=model)

            optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["learning_rate"]))
            num_training_steps = len(train_loader) * cfg["train"]["epochs"]
            scheduler = get_scheduler(
                "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
            )

            model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
                model, optimizer, train_loader, val_loader, scheduler
            )

            mt.train(model, optimizer, scheduler, train_loader, cfg)
            mt.evaluate_model(model, val_loader, tokenizer, cfg)
            mt.save_model(model, tokenizer, cfg)
            #upload_model_hub(model, tokenizer, cfg)

        logger.info("Machine Translation Workflow completed successfully.")
    except CustomException as e:
        logger.error(f"Execution failed: {e}")
        raise


if __name__ == "__main__":
    execute_machine_translation()
