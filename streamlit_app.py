import streamlit as st
import requests

# Mapping of task name to endpoint
endpoint_map = {
    "Text Generation": "/generate",
    "QA": "/qa",
    "Translation": "/translate",
    "Summarization": "/summarize",
    "Classification": "/classify"
}

# Select Task
task = st.selectbox("Select Task", list(endpoint_map.keys()))

# Dynamic input fields based on selected task
if task == "QA":
    question = st.text_input("Question")
    context = st.text_area("Context")
else:
    input_text = st.text_area("Input Text")

# Submit button
if st.button("Run"):
    endpoint = endpoint_map[task]

    # Prepare request payload
    if task == "QA":
        if not question or not context:
            st.warning("Please provide both question and context.")
        else:
            payload = {"question": question, "context": context}
    else:
        if not input_text:
            st.warning("Please enter input text.")
        else:
            payload = {"input_text": input_text}

    # Send request if payload is valid
    if "payload" in locals():
        try:
            res = requests.post(f"http://localhost:8000{endpoint}", json=payload)
            response = res.json()

            if "output" in response:
                st.write("Output:", response["output"])
            else:
                st.error("Server error: " + str(response.get("error", "Unknown error")))
        except Exception as e:
            st.error(f"Request failed: {e}")
