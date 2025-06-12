# TalentScout-Hiring-Assistant-Chatbot

**TalentScout Hiring Assistant** is a Streamlit-based AI chatbot designed to streamline the technical interview process. It greets job candidates, collects their basic details (name, email, phone, experience, position, location, tech stack), and then automatically generates **five personalized technical questions** tailored to the candidate’s declared tech stack, experience and based on job role. Candidates can answer these questions and even ask follow-up clarifications (by prefixing their input with `query:`). The system uses a large language model (LLM) to generate questions and a sentiment analysis model to evaluate each answer. At the end of the session, it displays an assessment summary of the candidate’s responses and allows downloading a PDF report of all Q\&A. Under the hood, the app is built with **Streamlit**, uses **LangChain** for managing LLM interactions, and Hugging Face’s **Transformers** library for running the models.

## Features

* **Candidate Greeting & Info Collection:** The app starts by greeting the candidate and asking them to fill out a simple form with their details (name, email, phone, years of experience, desired position, location, and technology stack). This ensures the chatbot has context about the candidate before proceeding.
* **Personalized Question Generation:** Using the candidate’s tech stack and position, the system invokes an LLM (specifically, the Qwen3-4B model) to generate five relevant technical interview questions. Each question is tailored to the technologies and role provided. For example, a candidate with a Python/Django stack might receive questions about Django REST framework or asynchronous programming. The generation relies on the capabilities of the Qwen LLM (an open-source model from Alibaba Cloud).
* **Clarification Queries:** After seeing a question, the candidate can type a clarification request by starting their input with `query:`. This triggers the chatbot to use the LLM again to produce a helpful follow-up explanation or hint related to that question. This lets the candidate ask for further details on any technical question.
* **Sentiment Analysis of Answers:** When the candidate submits an answer, the app uses a pre-trained **DistilBERT** model to analyze the sentiment of the response. DistilBERT is a smaller, faster variant of BERT (having about half the parameters of BERT base and still retaining about 95% of its performance). It is fine-tuned for sentiment analysis (binary positive/negative on SST-2). The sentiment score or label is recorded for each answer, giving a sense of the candidate’s tone or confidence. The sentiment analysis is implemented via Hugging Face’s “text-classification” pipeline (alias “sentiment-analysis”).
* **Assessment Summary & PDF Report:** After all five questions are answered, the app compiles the results into a summary. This includes each question, the candidate’s answer, and the detected sentiment. The summary is shown on-screen as a final assessment. Additionally, the app generates a downloadable PDF report containing all the questions and answers (using the ReportLab library), so the interviewer can review or archive the interview session. The PDF generation is handled by ReportLab, an open-source Python library for creating PDFs and graphics.

## Pipeline Structure

The app’s workflow proceeds as follows:

1. **Start & Greeting:** The Streamlit app (`app.py`) launches and displays a welcome message. It shows a sidebar or form where the candidate enters their details (name, contact info, experience, position, location, tech stack).
2. **Submit Details:** Once the candidate submits the form, the app captures this information. In the background, a LangChain **LLM Chain** is prepared with a system prompt that incorporates the candidate’s tech stack and role.
3. **Generate Questions:** The chain calls the Qwen3-4B LLM (via Hugging Face Transformers) to generate five technical questions. These questions are based on the candidate’s inputs (e.g. their specific programming languages or frameworks) to ensure relevance. The questions are displayed one at a time in the main interface.
4. **Answer Loop:** For each generated question:

   * The candidate types an answer into a text box.
   * If the candidate types a follow-up with the prefix `query:`, the app recognizes this as a clarification request. The chatbot then uses the LLM again (with the query appended to the context) to produce a helpful explanation.
   * After the final answer to that question, the app runs the answer text through the DistilBERT sentiment model. The resulting sentiment (positive/negative or score) is stored.
   * The app then proceeds to the next question.
5. **Final Summary:** After all questions are done, the app uses another LangChain chain (or simple string formatting) to compile an assessment. This includes all Q\&A pairs and sentiment results. The summary is displayed on-screen.
6. **PDF Download:** Concurrently, a PDF is generated with the Q\&A summary using ReportLab. A download button allows the candidate/interviewer to save this PDF report.

This end-to-end pipeline – from greeting to Q\&A to summary – is orchestrated within Streamlit’s flow, with LangChain handling the structured LLM calls and Transformers models for inference.

## Models

The project uses two main machine learning models:

* **Qwen3-4B (LLM for question generation):** This is a 4-billion-parameter open-source large language model in Alibaba Cloud’s Qwen (Tongyi Qianwen) family. Qwen models are built for tasks like coding and reasoning and support many languages. The Qwen3-4B variant is relatively lightweight but still very capable; in fact, it’s reported that even this “tiny” 4B model can rival larger versions in many tasks. We use Qwen3-4B via the Hugging Face Transformers library to generate contextual questions and answer clarifications.
* **DistilBERT (sentiment analysis):** We use the `distilbert-base-uncased-finetuned-sst-2-english` model from Hugging Face, which is a version of BERT distilled to be smaller and faster. DistilBERT has about half the parameters of the original BERT-base model and retains \~95% of its performance on language understanding benchmarks. It has been fine-tuned on the Stanford Sentiment Treebank (SST-2) dataset for binary sentiment classification. This model is used to classify each candidate’s answer as positive or negative (or provide a confidence score). The sentiment analysis is applied through Hugging Face’s pipeline interface, which abstracts away the tokenization and model call.

No other external ML services are required. All inference runs locally using PyTorch (the backend of Transformers) on the host machine (CPU or GPU, depending on availability).

## Tech Stack

The application is built entirely in **Python** using a collection of open-source libraries:

* **Streamlit:** A rapid development framework for Python web apps, used to build the interactive front-end. Streamlit allows turning Python scripts into web interfaces with widgets (buttons, forms, text areas) without having to write HTML/JS.
* **LangChain:** A framework for developing applications with large language models. LangChain is used here to structure the sequence of LLM calls (prompt templates, chains, memory). It manages the conversation flow between greeting, question generation, clarification, and summarization.
* **Hugging Face Transformers:** This library provides easy access to pre-trained LLMs and other NLP models. We use it to load and run the Qwen3-4B model and the DistilBERT model. Transformers handles tokenization and model inference behind the scenes.
* **PyTorch:** The primary deep learning backend for the model inference. Hugging Face’s Transformers will use PyTorch (a popular ML library) to run the neural network models on the CPU/GPU.
* **ReportLab:** A Python library for creating PDFs and graphics. It is used to assemble the final Q\&A report into a formatted PDF that the user can download.
* **Other libraries:** Standard Python libraries such as `streamlit` (for UI), `langchain` (for chains), `transformers` (for models), `torch` (for model runtime), `reportlab` (for PDF) and their dependencies are listed in the `requirements.txt`.

Together, these tools allow the app to collect input, call AI models, display results, and generate documents – all within a simple local application.

## Project Structure

- **`app.py`**: 
  - It handles the overall flow of the hiring assistant chatbot, including:
    - contains the Streamlit-based user interface.
    - Displaying the initial greeting and instructions.
    - Collecting candidate information (e.g., name, email, tech stack).
    - Triggering the generation of technical questions based on the candidate's tech stack.
    - Collecting and storing candidate answers.
    - Analyzing sentiment of candidate responses.
    - Displaying a summary of the conversation and providing a PDF download option for the assessment.
  - The script integrates with other modules (e.g., `chains.py`, `state.py`) to manage the conversation state and generate dynamic content.

- **`chains.py`**:
  - This module defines the LangChain pipelines used for various tasks in the chatbot.
  - It includes:
    - Prompt templates for different stages of the conversation (e.g., information gathering, technical question generation, closing remarks).
    - Chains that combine these prompts with the language model (LLM) to generate responses.
    - Specific chains for handling query relevance and question revision during the technical interview phase.
  - By separating the prompt engineering and model interactions, this file keeps the logic clean and reusable.

- **`model.py`**:
  - Responsible for loading and configuring the transformer models used in the project.
  - It sets up pipelines for:
    - Information gathering (using a lower temperature for more predictable responses).
    - Question generation (with higher creativity to produce varied technical questions).
    - Sentiment analysis (using a pre-trained model for analyzing candidate responses).
  - This modular approach ensures that model configurations are centralized and easy to update.

- **`prompts.py`**:
  - Contains all the prompt templates used in the chatbot.
  - These templates are designed to guide the language model in:
    - Collecting candidate information.
    - Generating technical questions tailored to the candidate's tech stack and experience.
    - Handling candidate queries and revising questions when necessary.
    - Providing closing remarks and summarizing the conversation.
  - Each prompt is carefully crafted to ensure clarity, relevance, and alignment with the chatbot's objectives.

- **`state.py`**:
  - Manages the state of the conversation using the `HiringState` class.
  - Tracks:
    - The current stage of the conversation (e.g., greeting, info gathering, technical interview).
    - Candidate data (e.g., name, email, tech stack).
    - Generated technical questions and candidate answers.
    - Conversation history and logs.
  - This class ensures that the chatbot maintains context throughout the interaction, providing a seamless experience for the candidate.

- **`utils.py`**:
  - Contains utility functions that support various operations in the chatbot, including:
    - Cleaning and formatting model responses.
    - Extracting questions from generated text.
    - Formatting the tech stack input.
    - Analyzing sentiment of candidate responses.
    - Generating a PDF of the technical assessment.
  - These helper functions keep the main logic in `app.py` and other modules clean and focused.

- **`conversation.py`**:
  - Handles the core conversation logic, including:
    - Managing the flow between different stages of the conversation.
    - Generating technical questions based on candidate data.
    - Processing candidate responses and queries.
    - Logging interactions and updating the conversation state.
  - This module acts as the brain of the chatbot, ensuring that the interaction is coherent and purposeful.

## Installation

To run the TalentScout Hiring Assistant locally, follow these steps:

1. **Clone the repository:** Download or clone the project code to your local machine.
2. **Install Python 3:** Ensure you have Python 3.8+ installed.
3. **Create a virtual environment (optional but recommended):**

   ```bash
   python3 -m venv env
   source env/bin/activate   # (on Windows use: env\Scripts\activate)
   ```
4. **Install dependencies:** Use the provided `requirements.txt` to install all required packages:

   ```bash
   pip install -r requirements.txt
   ```

   This will install Streamlit, LangChain, Transformers, torch, ReportLab, and other necessary libraries. For example, Streamlit itself is installed via pip (e.g. `pip install streamlit`).
5. **Run the app:** Launch the Streamlit app by running:

   ```bash
   streamlit run app.py
   ```

   This will open the application in your web browser (usually at `http://localhost:8501`). The Streamlit server is lightweight and should start up quickly.
6. **Use the app:** In the browser, fill out the candidate form and proceed through the chatbot interaction as described above. You do not need any API keys since the models run locally.

## Demo Link

*Video Link - https://www.loom.com/share/c47ff9e833ec45bb88b7c82572fbfa1b?sid=39e98c63-bce2-4ee9-ae44-980595d313e4


