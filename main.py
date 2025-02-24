import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY is missing. Please add it in your environment variables.")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# LangChain and document processing imports
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from operator import itemgetter
from langchain_core.runnables import chain
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Initialize FastAPI app
app = FastAPI()

# Enable CORS to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define a Pydantic model for chat requests
class ChatRequest(BaseModel):
    role: str
    message: str

# Initialize conversation memory using ChatOpenAI (passing the API key)
chat_memory = ConversationSummaryMemory(
    llm=ChatOpenAI(openai_api_key=openai_api_key),
    memory_key='message_log'
)

# Create vector stores for each role by loading corresponding PDF documents
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

for role, file_path in role_files.items():
    try:
        page = PyPDFLoader(file_path)
        role_document = page.load()
    except Exception as e:
        raise RuntimeError(f"Error loading file {file_path}: {e}")

    character_splitter = CharacterTextSplitter(separator=".", chunk_size=1000, chunk_overlap=100)
    character_splitted_documents = character_splitter.split_documents(role_document)

    # Clean up whitespace in document chunks
    for i in range(len(character_splitted_documents)):
        character_splitted_documents[i].page_content = ' '.join(character_splitted_documents[i].page_content.split())

    vector_stores[role] = Chroma.from_documents(
        embedding=embedding,
        documents=character_splitted_documents,
        persist_directory=f"./vector_store_{role}"
    )

# Define the prompt template
TEMPLATE = """
Answer the question strictly based on the provided context.
Avoid Hallucinating
If the answer in the dataset has it, the AI must always add Must Action: Activate Employee profile! Once you have created the employee profile the next step is to activate the profile. In the "Staff" tab click on "Not Activated" to view the employee profile which needs to be activated. Click on the "Send Activation E-mail Now". If the email address is added into the profile the employee will get a welcome email and the instruction to activate the profile. You can manully activate the employees' profile by clickin on the "Manually Activate All" button. If you are manually activting the staff make sure to create a password and a username for the staff membres.

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

# Function to extract "Must-Do" actions from context
def extract_must_do_actions(context):
    must_do_actions = []
    if 'Must-Do:' in context:
        for line in context.split('\n'):
            if 'Must-Do:' in line:
                must_do_actions.append(line.split('Must-Do:')[1].strip())
    return must_do_actions

# Intent detection to check if user is asking about adding an employee
def detect_intent(user_input):
    intents = {
        "add employee": [
            "add a new employee to my team?", "officially add", "setup a profile",
            "steps to add an employee", "steps to setup an account", "setup an account",
            "add an employee", "bring a new employee", "new employee", "register a new staff member",
            "register an employee", "officially add someone", "add a staff", "create an employee profile", "add a new hire"
        ]
    }
    for intent, phrases in intents.items():
        if any(phrase in user_input.lower() for phrase in phrases):
            return intent
    return "general"

# Core CoRAG chain: retrieve context and generate response
def corag_chain(user_input, user_role):
    retriever = vector_stores[user_role].as_retriever(search_type='mmr', search_kwargs={'k': 3, 'lambda_multi': 0.4})
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

    # Build and run the chain with conversation memory, prompt template, and chat model
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

    # Append must-do actions if available and the user has permission
    must_do_actions = extract_must_do_actions(context)
    if must_do_actions and user_role in roles_with_permission:
        response += "\n\n**Must-Do Action:** " + ', '.join(must_do_actions)

    chat_memory.save_context(inputs={"input": user_input}, outputs={"output": response})
    return response

# FastAPI health check endpoint
@app.get("/")
def read_root():
    return {"message": "FastAPI Chatbot is running."}

# FastAPI chat endpoint: expects JSON with a role and a message
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

# Run the FastAPI app using uvicorn, binding to 0.0.0.0 and the provided PORT (for Render/Codesandbox)
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
