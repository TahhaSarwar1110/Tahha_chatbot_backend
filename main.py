import re
import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# LangChain Libraries
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("❌ ERROR: OPENAI_API_KEY is missing.")

# Initialize FastAPI
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Define request model
class ChatRequest(BaseModel):
    role: str = ""  # Allow empty role initially
    message: str

# Load config
CONFIG_PATH = "chatbot_config.json"
if not os.path.exists(CONFIG_PATH):
    raise RuntimeError(f"❌ ERROR: Config file '{CONFIG_PATH}' not found.")
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# Initialize conversation memory with additional context for role
chat_memory = ConversationSummaryMemory(
    llm=ChatOpenAI(openai_api_key=openai_api_key),
    memory_key='message_log'
)

# Initialize embeddings
embedding = OpenAIEmbeddings(model="text-embedding-ada-002")

# Build vector stores dynamically
vector_stores = {}
for role, file_path in config["roles"].items():
    if not os.path.exists(file_path):
        print(f"⚠️ WARNING: '{file_path}' not found for role '{role}'. Skipping.")
        continue
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    splitter = CharacterTextSplitter(separator=".", chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(documents)
    for doc in split_docs:
        doc.page_content = ' '.join(doc.page_content.split())
    vector_stores[role] = Chroma.from_documents(
        embedding=embedding,
        documents=split_docs,
        persist_directory=f"{config['vector_store_dir']}/{role}"
    )

# Define prompt template
TEMPLATE = """
Precisely answer the question based on the provided context from the user's role-specific document.
If the answer is not in the context, respond: "Please reach out to your management for assistance."
Avoid hallucinating or guessing.
Only schedulers can assign positions. If any other role apart from Scheduler asks query related to assign a position, the bot should tell them that only Schedulers are allowed to assign positions.
If the user cannot find the option, or button, or add employee button or add employee option, the AI should respond: You may not have the necessary permissions.
The AI should ask for further details if the query is related to any error.

Current Conversation:
{message_log}

Question Context:
{context}

Human:
{question}

AI:
{must_action}
"""
prompt_template = PromptTemplate.from_template(TEMPLATE)

# Initialize ChatOpenAI
chat = ChatOpenAI(model="gpt-4-turbo", temperature=0.2, max_tokens=500, openai_api_key=openai_api_key)

# Extract role from conversation history
def get_stored_role(conversation_history: str) -> str:
    for line in conversation_history.split('\n'):
        if "Your role is set to" in line:
            return line.split("Your role is set to")[1].strip().replace(".", "").lower()
    return None

# Core response generation
def generate_response(user_input: str, user_role: str) -> str:
    memory_vars = chat_memory.load_memory_variables({})
    conversation_history = memory_vars.get('message_log', '')

    # Extract stored role if not provided
    stored_role = get_stored_role(conversation_history) if not user_role else user_role
    effective_role = stored_role.lower().strip() if stored_role else user_role.lower().strip()

    if effective_role not in vector_stores:
        response = f"Role '{effective_role}' is not supported. Please specify a valid role (e.g., admin, manager, accountant)."
        chat_memory.save_context(inputs={"input": user_input}, outputs={"output": response})
        return response

    # 🔹 Handle "assign employee to department"
    if "assign" in user_input.lower() and "department" in user_input.lower():
        if effective_role == "scheduler":
            retriever = vector_stores[effective_role].as_retriever(
                search_type='similarity_score_threshold', search_kwargs={'k': 3, 'score_threshold': 0.65}
            )
            retrieved_docs = retriever.invoke("Steps to assign an employee to a department")
            context = "\n".join([doc.page_content for doc in retrieved_docs])

            response = context if context.strip() else "The document does not contain steps for assigning an employee to a department. Please clarify your query."
        else:
            response = "There are three methods to assign an employee to a department:\n1. Using Department Assignment Button\n2. Through Detailed Employee Form\n3. Importing Department CSV File\nPlease specify which method you'd like to use."

        chat_memory.save_context(inputs={"input": user_input}, outputs={"output": response})
        return response

    # 🔹 Handle "assign position" logic
    if "assign" in user_input.lower() and ("position" in user_input.lower() or "role" in user_input.lower()):
        if effective_role == "scheduler":
            retriever = vector_stores[effective_role].as_retriever(
                search_type='similarity_score_threshold', search_kwargs={'k': 3, 'score_threshold': 0.65}
            )
            retrieved_docs = retriever.invoke("Steps to assign a position to an employee")
            context = "\n".join([doc.page_content for doc in retrieved_docs])

            response = context if context.strip() else "The document does not contain steps for assigning a position. Please clarify your query."
        else:
            response = "Only schedulers can assign positions. Please contact your scheduler for assistance."

        chat_memory.save_context(inputs={"input": user_input}, outputs={"output": response})
        return response

    response = "I’m not sure how to help with that yet. Could you provide more details?"
    chat_memory.save_context(inputs={"input": user_input}, outputs={"output": response})
    return response

# Endpoints
@app.get("/")
def read_root():
    return {"message": "FastAPI Chatbot is running."}

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    try:
        response = generate_response(request.message, request.role)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
