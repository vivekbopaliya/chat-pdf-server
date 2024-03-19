# Chat-PDF

Chat-PDF is an application designed to facilitate PDF uploads and answer questions related to them.

## Frontend Code

For the frontend React code, please refer [here](https://github.com/vivekbopaliya/chat-pdf-client).

## Installation (Backend)

### Prerequisites

- Python: Install Python from its original [site](https://www.python.org/downloads/).

### Setup Instructions

- Fork the Repository
  Fork the repository into your own GitHub account.

- Clone your newly forked repository from GitHub onto your local computer.

### Setup local environment

1. Run `python -m venv .venv` to create a virtual environment.
2. Download the dependencies mentioned in the `requirements.txt` file.

### Get OpenAI Key

- Obtain your own OpenAI key from [here](openai-key-link).

### Set Up Environment Variables

1. Create a `.env` file.
2. Set up your OpenAI key within it.

### Run the Application

- Run the command `uvicorn main:app --reload` to start the application.
- Navigate to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to test the APIs.

## APIs

Our application offers three APIs:

1. **PDF Upload API**: This API accepts a PDF file and sets up chains to answer questions related to the PDF content.

2. **Question Answering API**: With this API, you can submit a question, and utilizing the language-based knowledge chains established earlier which returns a suitable answer extracted from your PDF.

3. **PDF Retrieval API**: This API allows you to retrieve all the PDFs that have been uploaded. ( Note : this can be customized with each user and their pdfs but authentication is not the scope of this project. )

## Basic Architecture

### File Handling

- API accepts a file and validates if it's a PDF. If not, it returns a 400 error.

### File Processing

- The PDF file is read, and its binary content is converted into bytes using IO.

### Database Interaction

- Filename and filesize are stored in a PostgreSQL database for future retrieval.

### Text Extraction

- The content of pdf is extracted using FileReader from pypdf.

### Chunking Text

- The extracted text is divided into smaller chunks for efficient processing.

### Embedding Setup

- Embeddings are created from these chunks, establishing a chain to track the conversation.

### Semantic Search

- Semantic search is performed based on the user's question.

### Answer Retrieval

- Using OpenAI's language model, an appropriate answer is retrieved based on the semantic search results.
