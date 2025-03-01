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
If the answer is not in the context, respond: "Please reach out to your managment for the assistance"
Avoid hallucinating or guessing.
The AI should know that only schedulers can assign positions.
only schedulers can assign positions.
If the user cannot find the option or button, the AI should respond: You may not have the necessary permissions.
If the user cannot find the option or button, the AI should respond: You may not have the necessary permissions.
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

# Dynamic intent detection
def detect_intent(user_input: str) -> str:
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

# Check if user is specifying a method
def detect_method(user_input: str) -> str:
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

# Extract role from conversation history
def get_stored_role(conversation_history: str) -> str:
    for line in conversation_history.split('\n'):
        if "Your role is set to" in line:
            return line.split("Your role is set to")[1].strip().replace(".", "")
    return None

# Core response generation
def generate_response(user_input: str, user_role: str) -> str:
    # Load conversation history
    memory_vars = chat_memory.load_memory_variables({})
    conversation_history = memory_vars.get('message_log', '')

    # Check if this is the start of the conversation
    if not conversation_history and (not user_role or user_role not in vector_stores):
        response = "Hi! What’s your role? (e.g., admin, manager, accountant)"
        chat_memory.save_context(inputs={"input": user_input}, outputs={"output": response})
        return response

    # If role isn’t provided, check if it’s stored in memory
    stored_role = get_stored_role(conversation_history) if not user_role else user_role
    if not stored_role and user_role not in vector_stores:
        # User might be responding with their role
        potential_role = user_input.lower().strip()
        if potential_role in vector_stores:
            response = f"Your role is set to {potential_role}. How can I assist you now?"
            chat_memory.save_context(inputs={"input": user_input}, outputs={"output": response})
            return response
        else:
            response = "That’s not a valid role. Please specify a valid role (e.g., admin, manager, accountant)."
            chat_memory.save_context(inputs={"input": user_input}, outputs={"output": response})
            return response
    
    # Use stored role if user_role is empty but role was previously set
    effective_role = stored_role if not user_role else user_role
    if effective_role not in vector_stores:
        response = f"Role '{effective_role}' is not supported. Please specify a valid role (e.g., admin, manager, accountant)."
        chat_memory.save_context(inputs={"input": user_input}, outputs={"output": response})
        return response

    # Handle special cases


    visibility_triggers = ["dont see", "cant see", "not see", "cant find", "havent seen", "havent found"]
    add_employee_triggers = ["add employee", "employee button", "employee option"]
    permission_response = (
        "If you do not see the 'Add Employee' option or button, it means you do not have the necessary permission level. "
        "You must have manager, admin, supervisor, or scheduler access privileges in Humanity to add employees. "
        "Please contact an Admin or Manager.")
                                            
    for trigger, response in config.get("special_cases", {}).items():
        user_lower = user_input.lower()
        if re.search(r'\b' + re.escape(trigger) + r'\b', user_lower, re.IGNORECASE):
            # Special handling for visibility + add employee combos
            if any(v in user_lower for v in visibility_triggers) and any(a in user_lower for a in add_employee_triggers):
                chat_memory.save_context(inputs={"input": user_input}, outputs={"output": permission_response})
                return permission_response
            chat_memory.save_context(inputs={"input": user_input}, outputs={"output": response})
            return response

    # Detect intent
    intent = detect_intent(user_input)
    intent_config = config["intents"].get(intent, config["intents"]["general"])

    # Check for method specification in follow-up
    method = detect_method(user_input)
    if method and "add_employee" in conversation_history:
        retriever = vector_stores[effective_role].as_retriever(search_type='similarity_score_threshold', search_kwargs={'k': 3, 'score_threshold': 0.55})
        method_query = f"Steps to add an employee using {method}"
        retrieved_docs = retriever.invoke(method_query)
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        
        if not context.strip():
            response = "The provided document does not contain detailed steps for this method. Please clarify your query."
        else:
            must_action = "Activate Employee profile!\nOnce you have created the employee profile, the next step is to activate it. In the 'Staff' tab, click on 'Not Activated' to view the profile, then click 'Send Activation E-mail Now'. If an email address is provided, the employee will receive a welcome email with activation instructions. Alternatively, manually activate by clicking 'Manually Activate All' and ensure you create a username and password."
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

    # Handle intent with permissions
    if "permissions" in intent_config:
        if effective_role not in intent_config["permissions"]:
            response = f"As a {effective_role}, you do not have permission to {intent.replace('_', ' ')}. Please contact an Admin or Manager."
            chat_memory.save_context(inputs={"input": user_input}, outputs={"output": response})
            return response
        
        # Offer the three methods for permitted roles
        response = "There are three methods to add an employee:\n1. Using Employee Button\n2. Using Detailed Form\n3. Import CSV Files\nPlease specify which method you'd like to use."
        chat_memory.save_context(inputs={"input": user_input}, outputs={"output": response})
        return response
    
    # General intent: Use role-specific document context
    retriever = vector_stores[effective_role].as_retriever(search_type='similarity_score_threshold', search_kwargs={'k': 3, 'score_threshold': 0.65})
    retrieved_docs = retriever.invoke(user_input)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    
    if not context.strip():
        response = "The provided document does not contain this information. Please clarify your query."
    else:
        response_chain = (
            RunnablePassthrough.assign(
                message_log=RunnableLambda(lambda _: conversation_history),
                context=RunnablePassthrough(),
                must_action=RunnableLambda(lambda _: "")
            )
            | prompt_template
            | chat
            | StrOutputParser()
        )
        response = response_chain.invoke({
            "question": user_input,
            "context": context,
            "message_log": conversation_history
        })
    
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
