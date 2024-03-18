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

import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.post('/upload')
async def pdf_upload(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400, detail="Uploaded file must be a PDF")
    print(file)
    pdf_binary_data = await file.read()

    # converting unreadable binary file into bytes for pdfReader
    pdf = io.BytesIO(pdf_binary_data)

    pdf_content = PdfReader(pdf)

    text = ''
    for i in pdf_content.pages:
        text += i.extract_text()

    # # splitting file into chunks
    text_spiltter = CharacterTextSplitter(
        separator='\n',
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_spiltter.split_text(text)

    # Creating embedding from chunks
    load_dotenv()
    embedding = OpenAIEmbeddings()
    # embedding = HuggingFaceInstructEmbeddings(
    #     model_name="hkunlp/instructor-xl")

    # # setting up knowledge base
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embedding)
    llm = ChatOpenAI(temperature=0.5)
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl",
    #                      model_kwargs={"temperature": 0.5, "max_length": 512})
    print(llm)
    # memory = ConversationBufferMemory(
    #     memory_key='chat_history', return_messages=True)
    # conversation_chain = ConversationalRetrievalChain.from_llm(
    #     llm=llm,
    #     retriever=vectorstore.as_retriever(),
    #     memory=memory
    # )
    # print(conversation_chain)

    return {'msg': 'done'}
