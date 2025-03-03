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
from operator import itemgetter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# ------------------------------------
# Load Environment Variables
# ------------------------------------

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("❌ ERROR: OPENAI_API_KEY is missing.")

# ------------------------------------
# Initialize FastAPI
# ------------------------------------

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ------------------------------------
# Define Request Model
# ------------------------------------

class ChatRequest(BaseModel):
    role: str = ""  
    message: str

# ------------------------------------
# Load Configuration
# ------------------------------------

CONFIG_PATH = "chatbot_config.json"
if not os.path.exists(CONFIG_PATH):
    raise RuntimeError(f"❌ ERROR: Config file '{CONFIG_PATH}' not found.")

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# ------------------------------------
# Initialize Conversation Memory
# ------------------------------------

chat_memory = ConversationSummaryMemory(
    llm=ChatOpenAI(openai_api_key=openai_api_key),
    memory_key='message_log'
)

# ------------------------------------
# Initialize Embeddings
# ------------------------------------

embedding = OpenAIEmbeddings(model="text-embedding-ada-002")

# ------------------------------------
# Build Vector Stores Dynamically
# ------------------------------------

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

# ------------------------------------
# Define Prompt Template
# ------------------------------------

TEMPLATE = """
Precisely answer the question based on the provided context from the user's role-specific document.
If the answer is not in the context, respond: "Please reach out to your management for assistance."
Avoid hallucinating or guessing.
If the user cannot find an option or button, respond: "You may not have the necessary permissions."
If the query is related to any error, ask for further details.

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

# ------------------------------------
# Initialize Chat Model
# ------------------------------------

chat = ChatOpenAI(model="gpt-4-turbo", temperature=0.2, max_tokens=500, openai_api_key=openai_api_key)

# ------------------------------------
# Intent & Method Detection
# ------------------------------------

def detect_intent(user_input: str) -> str:
    """ Detects the most relevant intent based on input keywords. """
    user_input = user_input.lower().strip()
    best_intent = "general"
    highest_score = -1

    for intent, intent_config in config["intents"].items():
        if intent == "general":
            continue

        input_words = set(user_input.split())
        intent_keywords = set(intent_config["keywords"])
        required_keywords = set(intent_config.get("required", []))
        threshold = intent_config.get("threshold", 1)
        
        has_required = not required_keywords or bool(input_words & required_keywords)
        if not has_required:
            continue
        
        keyword_overlap = len(input_words & intent_keywords)
        if keyword_overlap >= threshold and keyword_overlap > highest_score:
            highest_score = keyword_overlap
            best_intent = intent
    
    return best_intent

def detect_method(user_input: str) -> str:
    """ Detects which method the user is selecting. """
    user_input = user_input.lower().strip()
    methods = {
        "employee button": ["employee button", "using employee button", "button"],
        "detailed form": ["detailed form", "using detailed form", "form"],
        "import csv": ["import csv", "csv", "import csv files", "csv files"]
    }

    for method, keywords in methods.items():
        if any(keyword in user_input for keyword in keywords):
            return method
    return None

def check_employee_method_selected(conversation_history: str) -> bool:
    """ Checks if the user has already selected a method to avoid repeating the question. """
    selected_methods = ["using employee button", "using detailed form", "import csv"]
    return any(method in conversation_history.lower() for method in selected_methods)

# ------------------------------------
# Core Response Logic
# ------------------------------------

def generate_response(user_input: str, user_role: str) -> str:
    """ Generates the chatbot's response based on intent detection and role-specific context. """
    memory_vars = chat_memory.load_memory_variables({})
    conversation_history = memory_vars.get('message_log', '')

    if "add_employee" in conversation_history and check_employee_method_selected(conversation_history):
        response = "You have already selected a method. If you need further assistance, let me know."
        chat_memory.save_context(inputs={"input": user_input}, outputs={"output": response})
        return response

    intent = detect_intent(user_input)

    method = detect_method(user_input)
    if method and "add_employee" in conversation_history:
        retriever = vector_stores[user_role].as_retriever(
            search_type='similarity_score_threshold', search_kwargs={'k': 3, 'score_threshold': 0.65}
        )

        method_query = f"Steps to add an employee using {method}"
        retrieved_docs = retriever.invoke(method_query)
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        if not context.strip():
            response = "I couldn't find specific steps for this method. Please clarify your query."
        else:
            must_action = "Activate Employee profile! Go to 'Staff' > 'Not Activated' and activate the profile."
            response_chain = (
                RunnablePassthrough.assign(
                    message_log=RunnableLambda(lambda _: conversation_history),
                    context=RunnablePassthrough(),
                    must_action=RunnableLambda(lambda _: must_action)
                )
                | prompt_template
                | chat
                | StrOutputParser()
            )

            response = response_chain.invoke({
                "question": method_query,
                "context": context,
                "message_log": conversation_history
            })

        chat_memory.save_context(inputs={"input": user_input}, outputs={"output": response})
        return response

    if intent == "add_employee":
        response = "There are three methods to add an employee:\n1. Using Employee Button\n2. Using Detailed Form\n3. Import CSV Files\nPlease specify which method you'd like to use."
        chat_memory.save_context(inputs={"input": user_input}, outputs={"output": response})
        return response

    return "I'm not sure how to help with that yet. Could you provide more details?"

# ------------------------------------
# API Endpoints
# ------------------------------------

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
