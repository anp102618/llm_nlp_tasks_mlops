import os
from huggingface_hub import create_repo, upload_folder, login
from pathlib import Path

# Optional: Use token from env variable (for GitHub Actions or CLI)
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print(" No token found in env. Using CLI login (requires huggingface-cli login).")

# Base path to your tuned models
BASE_DIR = Path("Tuned_models")
USERNAME = "anp102618"  # Replace with your actual Hugging Face username

def upload_model(task_folder: Path):
    task_name = task_folder.name
    repo_id = f"{USERNAME}/llm-{task_name}"

    print(f"Uploading '{task_name}' model to repo: {repo_id}")

    # Step 1: Create the repo (skip if exists)
    try:
        create_repo(repo_id, private=False, exist_ok=True)
    except Exception as e:
        print(f"Repo creation skipped or failed: {e}")

    # Step 2: Upload the model folder
    try:
        upload_folder(
            repo_id=repo_id,
            folder_path=str(task_folder),
            path_in_repo="",
            commit_message=f"Upload fine-tuned model for {task_name}"
        )
        print(f"Uploaded '{task_name}' successfully!\n")
    except Exception as e:
        print(f"Failed to upload {task_name}: {e}")

def execute_model_upload():
    if not BASE_DIR.exists():
        print(f"Base directory {BASE_DIR} not found.")
        return

    for task_folder in BASE_DIR.iterdir():
        if task_folder.is_dir():
            upload_model(task_folder)

if __name__ == "__main__":
    execute_model_upload()
