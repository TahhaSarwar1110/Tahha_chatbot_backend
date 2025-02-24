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

# Roles with permission to view must-do actions
roles_with_permission = ["admin", "manager", "supervisor", "scheduler"]

# Extract must-do actions from context
def extract_must_do_actions(context):
    must_do_actions = []
    if 'Must-Do:' in context:
        for line in context.split('\n'):
            if 'Must-Do:' in line:
                must_do_actions.append(line.split('Must-Do:')[1].strip())
    return must_do_actions

# Intent mapping for "add employee" queries
def detect_intent(user_input):
    intents = {
        "add employee": [
            "add a new employee to my team", "adding a new team member", "adding him", "adding her", "adding a new employee", "setup a new employee account", "setup his account", "setup her account", "officially add", "setup a profile", "register a new staff member", "correct way to add him", "correct way to add her",
            "steps to add an employee", "steps to setup an account", "creating a new account",  "setup an account", "give him system access", "add him", "input an employee", "input him", "input her",
            "add an employee", "add a new team member", "add a new user", "bring a new employee", "correct way to add an employee", "new employee", "register a new staff member", "input new team members", "input new hire",
            "register an employee","assign", "give system access", "officially add someone", "setting up", "add staff", "add a staff", "create an employee profile", "add a new hire"
        ]
    }
    for intent, phrases in intents.items():
        if any(phrase in user_input.lower() for phrase in phrases):
            return intent
    return "general"

#def check_mobile_query(user_input):
    #if "mobile" in user_input.lower():
        #return "Reach out to management"
    #return None  # Indicates no special handling needed

# Core CoRAG function that retrieves context and generates a response
def corag_chain(user_input, user_role):
    if user_role not in vector_stores:
        return f"Vector store for role '{user_role}' not available. Please check your role or document."
    retriever = vector_stores[user_role].as_retriever(search_type='similarity', search_kwargs={'k': 2})
    retrieved_docs = retriever.invoke(user_input)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    
    if not context.strip():
        return "The provided document does not contain this information. Please clarify your query."
    
    # Special handling for "add employee" intent
    if detect_intent(user_input) == "add employee":
        if user_role in roles_with_permission:
            return "There are multiple ways to do so. Would you like to use: 1) Use Employee Button, 2) Detailed, or 3) Bulk Upload?"
        else:
            return f"As a {user_role}, you do not have permission to add employees. Please contact an Admin or Manager."
    
    # Build the response chain using conversation memory, prompt template, and ChatOpenAI
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
    
    # Append must-do actions if available and permitted
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

# Run the FastAPI app (suitable for Render and CodeSandbox)
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
