import os
import logging
import torch
import yaml
import mlflow
import mlflow.pytorch
from datetime import datetime
from typing import Any, Dict, List, Tuple
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_scheduler,
    PreTrainedTokenizer,
    PreTrainedModel,
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

class Question_Answering(NLPBaseModel):
    def __init__(self) -> None:
        """Initializes the Question Answering model interface."""
        super().__init__()

    @track_performance
    def preprocess_function(self, examples: Dict[str, Any], tokenizer: PreTrainedTokenizer, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocesses the QA dataset by tokenizing and computing start/end positions.

        Args:
            examples (Dict[str, Any]): Batch of examples from the dataset.
            tokenizer (PreTrainedTokenizer): HuggingFace tokenizer.
            cfg (Dict[str, Any]): Configuration dictionary.

        Returns:
            Dict[str, Any]: Tokenized inputs with start and end positions.
        """
        try:
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            logger.info("Data preprocessing starting..")
            questions = [q.lstrip() for q in examples[cfg["dataset"]["question_column"]]]
            contexts = examples[cfg["dataset"]["context_column"]]
            inputs = tokenizer(
                questions,
                contexts,
                max_length=cfg["tokenizer"]["max_input_length"],
                truncation="only_second",
                padding=cfg["tokenizer"]["padding"],
                return_offsets_mapping=True,
                return_tensors=None,
            )

            offset_mapping = inputs.pop("offset_mapping")
            start_positions, end_positions = [], []

            for i, offsets in enumerate(offset_mapping):
                answer = examples[cfg["dataset"]["answer_column"]][i]
                start_char = answer["answer_start"][0] if isinstance(answer["answer_start"], list) else answer["answer_start"]
                end_char = start_char + len(answer["text"][0]) if isinstance(answer["text"], list) else len(answer["text"])

                sequence_ids = inputs.sequence_ids(i)
                context_start = sequence_ids.index(1)
                context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

                if not (offsets[context_start][0] <= start_char and offsets[context_end][1] >= end_char):
                    start_positions.append(0)
                    end_positions.append(0)
                else:
                    token_start_index = context_start
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    start_positions.append(token_start_index - 1)

                    token_end_index = context_end
                    while token_end_index >= 0 and offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    end_positions.append(token_end_index + 1)

            inputs["start_positions"] = start_positions
            inputs["end_positions"] = end_positions
            logger.info("Data preprocessing completed..")
            return inputs

        except CustomException as e:
            logger.error(f"Error in preprocessing: {e}")
            raise

    @track_performance
    def get_dataloaders(self, tokenizer: PreTrainedTokenizer, cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, Dataset]:
        """
        Loads and tokenizes the dataset, returning DataLoaders.

        Args:
            tokenizer (PreTrainedTokenizer): Tokenizer instance.
            cfg (Dict[str, Any]): Configuration dictionary.

        Returns:
            Tuple[DataLoader, DataLoader, Dataset]: Train/Validation DataLoaders and raw validation set.
        """
        try:
            logger.info("Data loading and tokenization starting..")
            dataset = load_dataset(cfg["dataset"]["name"])
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
            batch_size = cfg["train"]["batch_size"]

            train_loader = DataLoader(tokenized_train, batch_size=batch_size, collate_fn=collator, shuffle=True)
            val_loader = DataLoader(tokenized_val, batch_size=batch_size, collate_fn=collator)

            logger.info("Dataloaders created successfully.")
            return train_loader, val_loader, val_ds

        except CustomException as e:
            logger.error(f"Error creating dataloaders: {e}")
            raise

    @track_performance
    def get_model(self, cfg: Dict[str, Any]) -> PreTrainedModel:
        """
        Loads the model and applies LoRA configuration.

        Args:
            cfg (Dict[str, Any]): Configuration dictionary.

        Returns:
            PreTrainedModel: Model ready for training with LoRA.
        """
        try:
            logger.info("Model import starting..")
            model = AutoModelForQuestionAnswering.from_pretrained(
                cfg["model"]["name"],
                load_in_4bit=cfg["model"]["load_in_4bit"],
                torch_dtype=getattr(torch, cfg["model"]["torch_dtype"]),
                device_map=cfg["model"]["device_map"],
            )
            model = prepare_model_for_kbit_training(model)

            peft_config = LoraConfig(
                task_type=TaskType[cfg["lora"]["task_type"]],
                r=cfg["lora"]["r"],
                lora_alpha=cfg["lora"]["alpha"],
                lora_dropout=cfg["lora"]["dropout"],
                bias=cfg["lora"]["bias"],
                target_modules=cfg["model"]["target_modules"],
            )

            model = get_peft_model(model, peft_config)
            logger.info("Model loaded and prepared with QLoRA.")
            return model

        except CustomException as e:
            logger.error(f"Error loading model: {e}")
            raise

    @track_performance
    def train(self, model: PreTrainedModel, optimizer: torch.optim.Optimizer, scheduler: Any, train_loader: DataLoader, cfg: Dict[str, Any]) -> None:
        """
        Trains the model for question answering.

        Args:
            model (PreTrainedModel): The transformer model.
            optimizer (Optimizer): Optimizer.
            scheduler: Learning rate scheduler.
            train_loader (DataLoader): Training dataloader.
            cfg (Dict[str, Any]): Configuration dictionary.
        """
        try:
            logger.info("Model training starting..")
            for epoch in range(cfg["train"]["epochs"]):
                model.train()
                total_loss = 0
                loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']}")
                for batch in loop:
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        start_positions=batch["start_positions"],
                        end_positions=batch["end_positions"]
                    )
                    loss = outputs.loss
                    accelerator.backward(loss)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    total_loss += loss.item()
                    loop.set_postfix(loss=loss.item())
                mlflow.log_metric("train_loss", total_loss / len(train_loader), step=epoch)
            logger.info("Model training completed successfully..")

        except CustomException as e:
            logger.error(f"Training failed: {e}")
            raise

    @track_performance
    def evaluate_model(self, model: PreTrainedModel, val_loader: DataLoader, tokenizer: PreTrainedTokenizer,cfg: Dict[str, Any], 
                        raw_val_ds: Dataset, batch_size: int) -> Dict[str, float]:
        """
        Evaluates the model on the validation dataset.

        Args:
            model (PreTrainedModel): Trained model.
            val_loader (DataLoader): Validation DataLoader.
            tokenizer (PreTrainedTokenizer): Tokenizer.
            cfg (Dict[str, Any]): Configuration dictionary.
            raw_val_ds (Dataset): Raw validation dataset for original answers.
            batch_size (int): Batch size used during validation.

        Returns:
            Dict[str, float]: Evaluation metric results.
        """
        try:
            logger.info("Model evaluation starting...")
            metric = evaluate.load(cfg["evaluation"]["metric"])
            model.eval()

            predictions, references = [], []

            for i, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
                with torch.no_grad():
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"]
                    )

                start_logits = outputs.start_logits.cpu().numpy()
                end_logits = outputs.end_logits.cpu().numpy()
                input_ids = batch["input_ids"].cpu().numpy()

                for j in range(len(input_ids)):
                    start_idx = start_logits[j].argmax()
                    end_idx = end_logits[j].argmax()

                    if end_idx < start_idx:
                        pred_text = ""
                    else:
                        pred_text = tokenizer.decode(input_ids[j][start_idx:end_idx + 1], skip_special_tokens=True)

                    raw_ex = raw_val_ds[i * batch_size + j]

                    predictions.append({
                        "id": raw_ex["id"],
                        "prediction_text": pred_text
                    })

                    references.append({
                        "id": raw_ex["id"],
                        "answers": raw_ex["answers"]
                    })

            results = metric.compute(predictions=predictions, references=references)

            for k, v in results.items():
                mlflow.log_metric(f"eval_{k}", v)
            logger.info("Model evaluation completed successfully.")
            return results

        except CustomException as e:
            logger.error(f"Evaluation failed: {e}")
            raise

    @track_performance
    def save_model(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, cfg: Dict[str, Any]) -> None:
        """
        Saves the model and tokenizer to disk and logs to MLflow.

        Args:
            model (PreTrainedModel): Trained model.
            tokenizer (PreTrainedTokenizer): Corresponding tokenizer.
            cfg (Dict[str, Any]): Configuration dictionary.
        """
        try:
            logger.info("Saving model and tokenizer locally...")
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)

            output_dir = cfg["save"]["output_dir"]
            unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
            tokenizer.save_pretrained(output_dir)

            logger.info("Logging model and tokenizer artifacts to MLflow...")

            # Log model artifacts under 'model/' folder in MLflow
            for root, _, files in os.walk(output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    mlflow.log_artifact(file_path, artifact_path="model")

            # Log config file under 'config/' if it exists
            config_path = cfg.get("config_path", "config.yaml")
            if os.path.exists(config_path):
                mlflow.log_artifact(config_path, artifact_path="config")

            logger.info("Model, tokenizer, and config successfully logged to MLflow.")
        except CustomException as e:
            logger.error(f"Saving model/tokenizer failed: {e}")
            raise


@track_performance
def execute_qa() -> None:
    """
    Executes the complete Question Answering workflow:
    - Loads config and tokenizer
    - Loads model with LoRA
    - Prepares dataloaders
    - Trains, evaluates, and saves the model and artifacts
    """
    try:
        os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")
        logger.info("Commencing QA workflow..")
        cfg: Dict[str, Any] = load_config("./Config_Yaml/config_qa.yaml")
        mlflow.set_tracking_uri("https://dagshub.com/anp102618/llm_nlp_tasks_mlops.mlflow")
        experiment_name = cfg["mlflow"]["experiment_name"]
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        current_model = f"{cfg['mlflow']['experiment_name']}_{timestamp}"
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
            qa = Question_Answering()
            model = qa.get_model(cfg)
            train_loader, val_loader, raw_val_ds = qa.get_dataloaders(tokenizer, cfg)

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

            qa.train(model, optimizer, scheduler, train_loader, cfg)
            qa.evaluate_model(model, val_loader, tokenizer, cfg, raw_val_ds, cfg["train"]["batch_size"])
            qa.save_model(model, tokenizer, cfg)
            #upload_model_hub(model, tokenizer, cfg)
            logger.info("QA task completed successfully ..")

    except CustomException as e:
        logger.error(f"Execution failed: {e}")
        raise


if __name__ == "__main__":
    execute_qa()
