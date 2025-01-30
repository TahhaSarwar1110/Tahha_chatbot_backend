from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

# LangChain Libraries
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from operator import itemgetter
from langchain_core.runnables import chain

# RAG setup libraries
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

# Load environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load and process document
try:
    page = PyPDFLoader("User Query.pdf")
    my_document = page.load()

    character_splitter = CharacterTextSplitter(separator=".", chunk_size=500, chunk_overlap=50)
    character_splitted_documents = character_splitter.split_documents(my_document)

    for i in range(len(character_splitted_documents)):
        character_splitted_documents[i].page_content = " ".join(character_splitted_documents[i].page_content.split())

    # Initialize vector store and retriever
    embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
    vector_store = Chroma.from_documents(
        embedding=embedding,
        documents=character_splitted_documents,
        persist_directory="./TCP_directory_1"
    )
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3, "lambda_multi": 0.5})

except Exception as e:
    print(f"❌ ERROR LOADING DOCUMENTS: {e}")
    retriever = None  # Set to None if loading fails

# Initialize chatbot memory
chat = ChatOpenAI(
    model="gpt-4",
    temperature=0,
    max_tokens=250,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

chat_memory = ConversationSummaryMemory(llm=chat, memory_key="message_log")

# Define chatbot request model
class ChatRequest(BaseModel):
    message: str

# Define chatbot response template
TEMPLATE = """
The AI should completely understand the question and only answer strictly based on the provided document context.
The AI should not talk about the document in the answer.
The AI should not hallucinate.
If the question does not have enough information, the AI should ask for more details. Example Question: I want to edit the shifts for the whole week. Response: I will be able to assist with further details.
The AI should deliver well-structured, polished responses that enhance the overall user experience.
If the answer is not available in the context, the AI truthfully responds: "Sorry, I don't know the answer."
The AI when asked about deleting the availability should answer: To delete your availability, go to the availability module and click on it. Then, select the option to delete your availability.

Current Conversation:
{message_log}

Question Context:
{context}

Human:
{question}

AI:
"""
prompt_template = PromptTemplate.from_template(template=TEMPLATE)

# Define chatbot function
@chain
def memory_rag_chain(question):
    if retriever is None:
        return "Error: Document retrieval system is not available."

    retrieved_docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    if not context.strip():
        response = "Sorry, I don't know the answer."
        chat_memory.save_context(inputs={"input": question}, outputs={"output": response})
        return response

    chain = (
        RunnablePassthrough.assign(
            message_log=RunnableLambda(chat_memory.load_memory_variables) | itemgetter("message_log"),
            context=RunnablePassthrough()
        )
        | prompt_template
        | chat
        | StrOutputParser()
    )

    response = chain.invoke({"question": question, "context": context})
    chat_memory.save_context(inputs={"input": question}, outputs={"output": response})
    return response

# API Endpoint for chatbot
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        response = memory_rag_chain.invoke(request.message)
        return {"response": response}
    except Exception as e:
        print(f"❌ ERROR PROCESSING REQUEST: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
