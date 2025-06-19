from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import pipeline
import yaml
import os

# ---- Load YAML Config ----
with open("Tuned_Models/config.yaml") as f:
    config = yaml.safe_load(f)

TASK_MAP = {
    "text_generation": "text-generation",
    "text_classification": "text-classification",
    "summarization": "summarization",
    "question_answering": "question-answering",
    "machine_translation": "translation"
}
# ---- Initialize Pipelines ----
pipelines = {}
for task, data in config["models"].items():
    path = data["path"]
    hf_task = TASK_MAP.get(task)
    if not hf_task:
        raise ValueError(f"Invalid task in config: {task}")
    pipelines[task] = pipeline(hf_task, model=path, tokenizer=path)

# ---- API App ----
app = FastAPI()

# ---- Request Schema ----
class InferenceRequest(BaseModel):
    input_text: str


class QARequest(BaseModel):
    question: str
    context: str

# ---- Endpoints ----
@app.post("/generate")
def generate(req: InferenceRequest):
    output = pipelines["text_generation"](req.input_text, max_length=50)
    return {"output": output}

@app.post("/qa")
def qa(req: QARequest):
    output = pipelines["question_answering"](question=req.question, context=req.context)
    return {"output": output}

@app.post("/translate")
def translate(req: InferenceRequest):
    output = pipelines["machine_translation"](req.input_text)
    return {"output": output}

@app.post("/summarize")
def summarize(req: InferenceRequest):
    output = pipelines["summarization"](req.input_text)
    return {"output": output}

@app.post("/classify")
def classify(req: InferenceRequest):
    output = pipelines["text_classification"](req.input_text)
    return {"output": output}
