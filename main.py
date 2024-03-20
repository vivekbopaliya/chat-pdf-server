import io
from pypdf import PdfReader
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import psycopg2

app = FastAPI()


# to prevent CORS error
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

load_dotenv()

# Connecting with PostgreSQL


# Bytes to KB conversion as we are stroing this in PostgreSQL (e.g. "420.05 KB")
# def bytes_to_kilobytes(bytes_value):
#     return f"{bytes_value / 1024:.2f} KB"


# Declaring this globally as we will have to use this in 2 endpoints
knowledge_base = None


@app.get('/')
def index():
    return {"msg": 'It is working...'}


@app.post('/upload')
async def pdf_upload(file: UploadFile = File(...)):
    global knowledge_base

    # Return errors for non-PDF files
    if not file:
        return Response({'error': 'Please upload a file'}, status_code=400)
    if not file.filename.endswith('.pdf'):
        return Response({'error': 'Uploaded file must be a PDF'}, status_code=401)
    try:

        pdf_binary_data = await file.read()

        # converting unreadable binary file into bytes for pdfReader
        pdf = io.BytesIO(pdf_binary_data)
        pdf_content = PdfReader(pdf)

        # Fetching content from pdf
        text = ''
        for i in pdf_content.pages:
            text += i.extract_text()

        # splitting content into chunks to perform semantic search
        text_spiltter = CharacterTextSplitter(
            separator='\n',
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_spiltter.split_text(text)

        # Creating embedding from chunks
        embeddings = OpenAIEmbeddings()

        # Setting up knowledge base based on chunks
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # If everything works correctly, we store the file metadata in PostgreSQL
        if knowledge_base:
            # bytes_value = file.size
            # kilobytes_value = bytes_to_kilobytes(bytes_value)
            # cursor.execute(
            #     "INSERT INTO  file(file_name, file_size) VALUES (%s, %s)",
            #     (str(file.filename), kilobytes_value)
            # )
            # connection.commit()
            return {'msg': 'Knowledge base set up successfully'}

        return HTTPException(status_code=500, detail='Something went wrong settting up knowledge base.')

    except Exception as error:
        return HTTPException(status_code=500, detail=f'There was an error in server side {error}')


class Question(BaseModel):
    question: str


# ^^ for some reason just question:'str' was causing an error
@app.post('/chat')
async def question_and_answer(question: Question):
    global knowledge_base
    try:
        # Performing similarity search based upo user's question
        if question.question and knowledge_base:
            docs = knowledge_base.similarity_search(question.question)

            # Using LLM model to setup the chain of the knowledge and retrive the appropriate answer
            llm = OpenAI(model="gpt-3.5-turbo-instruct")
            chain = load_qa_chain(llm, chain_type="stuff")

            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs,
                                     question=question)
                print(cb)
            # Returning the answer
            return response
        return HTTPException(status_code=401, detail='Please upload a PDF before you ask questions.')
    except Exception as error:
        return HTTPException(status_code=500, detail=f'There was error on server side {error}')


# @app.get('/pdfs')
# def get_pdfs():
#     try:
#         cursor.execute('SELECT * FROM file')
#         files = cursor.fetchall()
#         return {'files': files}

#     except Exception as error:
#         return HTTPException(status_code=500, detail=error)
