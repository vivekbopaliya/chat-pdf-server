from fastapi import FastAPI, UploadFile, File, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
import psycopg2
import io

app = FastAPI()


# to prevent CORS error
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# Connecting with PostgreSQL
connection = psycopg2.connect(
    dbname='chat-pdf',
    user='postgres',
    password='VBGamer07',
    host='localhost',
    port='5432'
)

cursor = connection.cursor()


# Bytes to KB conversion (e.g. "420.05 KB")
def bytes_to_kilobytes(bytes_value):
    return f"{bytes_value / 1024:.2f} KB"


@app.post('/upload')
async def pdf_upload(file: UploadFile = File(...)):

    # Return 400 error for non-PDF files
    if not file:
        return Response({'msg': 'Please upload a file'}, status_code=400)
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400, detail="Uploaded file must be a PDF")

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
    if chunks:
        bytes_value = file.size
        kilobytes_value = bytes_to_kilobytes(bytes_value)
        cursor.execute(
            "INSERT INTO  file(file_name, file_size) VALUES (%s, %s)",
            (str(file.filename), kilobytes_value)
        )
        connection.commit()
    return {'msg': 'done'}
    # Creating embedding from chunks
    # load_dotenv()
    # embedding = OpenAIEmbeddings()
    # embedding = HuggingFaceInstructEmbeddings(
    #     model_name="hkunlp/instructor-xl")

    # # setting up knowledge base
    # vectorstore = FAISS.from_texts(texts=chunks, embedding=embedding)
    # llm = ChatOpenAI(temperature=0.5)
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl",
    #                      model_kwargs={"temperature": 0.5, "max_length": 512})
    # print(llm)
    # memory = ConversationBufferMemory(
    #     memory_key='chat_history', return_messages=True)
    # conversation_chain = ConversationalRetrievalChain.from_llm(
    #     llm=llm,
    #     retriever=vectorstore.as_retriever(),
    #     memory=memory
    # )
    # print(conversation_chain)

    # return {'msg': 'done'}


@app.get('/pdfs')
def get_pdfs():
    try:
        cursor.execute('SELECT * FROM file')
        files = cursor.fetchall()
        print(files)
        return {'files': files}
        # file_records = [{'id': row[0], 'file_name': row[1],
        #                  'file_size': row[2]} for row in files]

        # print(file_records)
        # return {'files': file_records}
    except Exception as error:
        return Response({'error': error}, status_code=500)
