import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# LangChain Libraries
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, chain
from operator import itemgetter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("❌ ERROR: OPENAI_API_KEY is missing. Add it in your environment variables.")

# Initialize FastAPI
app = FastAPI()

# Enable CORS to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request model that expects both role and message
class ChatRequest(BaseModel):
    role: str
    message: str

# Initialize conversation memory using ChatOpenAI with API key
chat_memory = ConversationSummaryMemory(
    llm=ChatOpenAI(openai_api_key=openai_api_key),
    memory_key='message_log'
)

# Define vector stores for multiple roles
vector_stores = {}
embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
role_files = {
    "accountant": 'Accountant.pdf',
    "admin": 'Admin.pdf',
    "employee with scheduling(individual managment)rights": 'Employee with Scheduling(Individual Managment)Rights.pdf',
    "employee": 'Employee.pdf',
    "manager": 'Manager.pdf',
    "scheduler": 'Scheduler.pdf',
    "schedule viewer": 'Schedule viewer.pdf',
    "supervisor": 'Supervisor.pdf'
}

# Build vector store for each role
for role, file_path in role_files.items():
    if not os.path.exists(file_path):
        print(f"⚠️ WARNING: '{file_path}' not found for role '{role}'. Skipping this role.")
        continue
    page = PyPDFLoader(file_path)
    role_document = page.load()
    character_splitter = CharacterTextSplitter(separator=".", chunk_size=1000, chunk_overlap=100)
    character_splitted_documents = character_splitter.split_documents(role_document)
    for i in range(len(character_splitted_documents)):
        character_splitted_documents[i].page_content = ' '.join(character_splitted_documents[i].page_content.split())
    vector_stores[role] = Chroma.from_documents(
        embedding=embedding,
        documents=character_splitted_documents,
        persist_directory=f"./vector_store_{role}"
    )

# Define prompt template
TEMPLATE = """
Precisely Answer the question strictly based on the provided context.
The AI should not make wild guesses.
Avoid Hallucinating

Current Conversation:
{message_log}

Question Context:
{context}

Human:
{question}

AI:
"""
prompt_template = PromptTemplate.from_template(template=TEMPLATE)

# Initialize ChatOpenAI instance for generating responses
chat = ChatOpenAI(
    model="gpt-4-turbo",
    temperature=0,
    max_tokens=500,
    openai_api_key=openai_api_key
)

# Normalize role keys for matching (lowercase keys)
normalized_roles = {role.replace("_", " ").lower(): role for role in role_files.keys()}

# Roles with permission to view must-do actions and add employees
roles_with_permission = ["admin", "manager", "supervisor", "scheduler"]

# Define methods and instructions for adding employees
add_employee_methods = {
    "use employee button": {
        "description": "Use Employee Button",
        "steps": "Click on the 'Employee' button in the dashboard, then select 'Add New' and fill in the basic details like name and email."
    },
    "detailed": {
        "description": "Detailed",
        "steps": "1. Click on the 'Staff' module from the top panel. 2. Select 'Add Employees' at the top right. 3. Choose 'Detailed Form' tab. 4. Fill in required fields (first name, last name, email). 5. Check 'Send Activation' box to email login instructions. Must Action: Activate the profile via 'Not Activated' tab and send activation email or manually activate with a username/password."
    },
    "bulk upload": {
        "description": "Bulk Upload",
        "steps": "Go to the 'Staff' module, select 'Add Employees,' choose 'Bulk Upload' tab, upload a CSV file with employee details, and confirm the import."
    }
}

# Extract must-do actions from context
def extract_must_do_actions(context):
    must_do_actions = []
    if 'Must-Do:' in context:
        for line in context.split('\n'):
            if 'Must-Do:' in line:
                must_do_actions.append(line.split('Must-Do:')[1].strip())
    return must_do_actions

# Intent detection for "add employee" and method selection
def detect_intent(user_input):
    intents = {
        "add employee": [
            "add a new employee to my team", "adding a new team member", "adding him", "adding her", "adding a new employee", "setup a new employee account", "setup his account", "setup her account", "officially add", "setup a profile", "register a new staff member", "correct way to add him", "correct way to add her",
            "steps to add an employee", "steps to setup an account", "creating a new account", "setup an account", "give him system access", "add him", "input an employee", "input him", "input her",
            "add an employee", "add a new team member", "add a new user", "bring a new employee", "correct way to add an employee", "new employee", "register a new staff member", "input new team members", "input new hire",
            "register an employee", "assign", "give system access", "officially add someone", "setting up", "add staff", "add a staff", "create an employee profile", "add a new hire"
        ]
    }
    user_input_lower = user_input.lower()
    
    # Check for "add employee" intent (general, not mobile-specific)
    for intent, phrases in intents.items():
        if any(phrase in user_input_lower for phrase in phrases) and "mobile" not in user_input_lower:
            return intent
    
    # Check for mobile-specific "add employee" intent
    if any(phrase in user_input_lower for phrase in intents["add employee"]) and "mobile" in user_input_lower:
        return "add employee mobile"
    
    # Check for method selection (e.g., "detailed", "1", "use employee button")
    method_keys = add_employee_methods.keys()
    method_descriptions = [method["description"].lower() for method in add_employee_methods.values()]
    number_options = ["1", "2", "3"]  # Maps to methods in order
    
    if any(method in user_input_lower for method in method_keys) or \
       any(desc in user_input_lower for desc in method_descriptions) or \
       user_input_lower in number_options:
        return "method_selection"
    
    return "general"

# Core CoRAG function that retrieves context and generates a response
def corag_chain(user_input, user_role):
    if user_role not in vector_stores:
        return f"Vector store for role '{user_role}' not available. Please check your role or document."
    
    retriever = vector_stores[user_role].as_retriever(search_type='similarity', search_kwargs={'k': 2})
    retrieved_docs = retriever.invoke(user_input)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    
    if not context.strip():
        return "The provided document does not contain this information. Please clarify your query."
    
    # Get intent
    intent = detect_intent(user_input)
    
    # Handle general "add employee" intent
    if intent == "add employee":
        if user_role in roles_with_permission:
            method_options = ", ".join(f"{i+1}) {method['description']}" for i, method in enumerate(add_employee_methods.values()))
            response = f"There are multiple ways to add an employee. Please select a method: {method_options}."
            chat_memory.save_context(inputs={"input": user_input}, outputs={"output": response})
            return response
        else:
            response = f"As a {user_role}, you do not have permission to add employees. Please contact an Admin or Manager."
            chat_memory.save_context(inputs={"input": user_input}, outputs={"output": response})
            return response
    
    # Handle mobile-specific "add employee" intent
    elif intent == "add employee mobile":
        if user_role in roles_with_permission:
            # Check if context contains mobile-specific info
            if "mobile" in context.lower():
                response_chain = (
                    RunnablePassthrough.assign(
                        message_log=(RunnableLambda(lambda inputs: chat_memory.load_memory_variables(inputs)) | itemgetter("message_log")),
                        context=RunnablePassthrough()
                    )
                    | prompt_template
                    | chat
                    | StrOutputParser()
                )
                response = response_chain.invoke({
                    "question": user_input,
                    "context": context,
                    "message_log": chat_memory.load_memory_variables({}).get('message_log', '')
                })
            else:
                response = "Sorry, I don’t have information on adding an employee using a mobile app based on the provided documents. Please try a general query like 'How do I add an employee?' or consult your system’s mobile app documentation."
            chat_memory.save_context(inputs={"input": user_input}, outputs={"output": response})
            return response
        else:
            response = f"As a {user_role}, you do not have permission to add employees. Please contact an Admin or Manager."
            chat_memory.save_context(inputs={"input": user_input}, outputs={"output": response})
            return response
    
    # Handle method selection
    elif intent == "method_selection":
        user_input_lower = user_input.lower()
        method_selected = None
        
        # Map user input to a method
        for i, (method_key, method_data) in enumerate(add_employee_methods.items()):
            if method_key in user_input_lower or \
               method_data["description"].lower() in user_input_lower or \
               str(i + 1) == user_input_lower:
                method_selected = method_key
                break
        
        if method_selected:
            response = add_employee_methods[method_selected]["steps"]
            chat_memory.save_context(inputs={"input": user_input}, outputs={"output": response})
            return response
        else:
            response = "I didn’t understand your selection. Please choose a method like 'Detailed', 'Use Employee Button', 'Bulk Upload', or a number (1, 2, 3)."
            chat_memory.save_context(inputs={"input": user_input}, outputs={"output": response})
            return response
    
    # General case (non-"add employee" queries)
    response_chain = (
        RunnablePassthrough.assign(
            message_log=(RunnableLambda(lambda inputs: chat_memory.load_memory_variables(inputs)) | itemgetter("message_log")),
            context=RunnablePassthrough()
        )
        | prompt_template
        | chat
        | StrOutputParser()
    )
    
    response = response_chain.invoke({
        "question": user_input,
        "context": context,
        "message_log": chat_memory.load_memory_variables({}).get('message_log', '')
    })
    
    must_do_actions = extract_must_do_actions(context)
    if must_do_actions and user_role in roles_with_permission:
        response += "\n\n**Must-Do Action:** " + ', '.join(must_do_actions)
    
    chat_memory.save_context(inputs={"input": user_input}, outputs={"output": response})
    return response

# Health check endpoint
@app.get("/")
def read_root():
    return {"message": "FastAPI Chatbot is running."}

# Chat endpoint that accepts a JSON payload with role and message
@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    user_role_key = request.role.lower()
    if user_role_key in normalized_roles:
        user_role = normalized_roles[user_role_key]
    else:
        raise HTTPException(status_code=400, detail="Invalid role provided.")
    
    try:
        response = corag_chain(request.message, user_role)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
