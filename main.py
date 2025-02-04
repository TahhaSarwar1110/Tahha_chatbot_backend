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

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise RuntimeError("❌ ERROR: OPENAI_API_KEY is missing. Add it in Render's Environment Variables.")

# Initialize FastAPI
app = FastAPI()

# Enable CORS to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all frontend origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Root Route (For Health Check)
@app.get("/")
def home():
    return {"message": "FastAPI Chatbot is running successfully!"}

# Load and process document
retriever = None  # Default to None in case of failure
try:
    pdf_path = "User Query.pdf"

    if not os.path.exists(pdf_path):
        print(f"⚠️ WARNING: '{pdf_path}' not found! Continuing without document retrieval.")
    else:
        page = PyPDFLoader(pdf_path)
        my_document = page.load()

        character_splitter = CharacterTextSplitter(separator=".", chunk_size=500, chunk_overlap=50)
        character_splitted_documents = character_splitter.split_documents(my_document)

        for doc in character_splitted_documents:
            doc.page_content = " ".join(doc.page_content.split())

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

# Initialize chatbot memory
chat = ChatOpenAI(
    model="gpt-4",
    temperature=0,
    max_tokens=250,
    openai_api_key=openai_api_key
)

chat_memory = ConversationSummaryMemory(llm=chat, memory_key="message_log")

# Define chatbot request model
class ChatRequest(BaseModel):
    message: str

# Define chatbot response template
TEMPLATE = """
The AI should:
- **Strictly answer based on the provided document context** and never hallucinate.
- **Fully understand the user's question** before answering.
- **Provide detailed, structured, and polished responses** to enhance user experience.
- **Avoid discussing the document** itself in the answer.
- **Ask for clarification** if a question is vague or lacks details.
- **Suggest contacting management** if a query is completely out of context.
- **Handle common user queries** with pre-defined responses to maintain consistency.

**Handling vague questions:**
If a question is too broad or unclear, the AI must request clarification instead of making assumptions.  
- Example: **"I need help with overtime settings."**  
  - ✅ Expected: *"Could you clarify what aspect of overtime settings you need help with? Are you configuring daily overtime, weekly overtime, or thresholds for specific employees?"*  
  - ❌ Failure: If it assumes the user wants to set an 8-hour threshold for an employee.

**Example Clarifications & Responses:**
1. **Updating emails in bulk** → "Bulk edit feature does not allow users to update employee emails in bulk. It only allows adding employees in bulk."
2. **Reactivating a permanent employee's email** → "To deactivate the temporary email associated with the employee, go to Staff > Disabled > the employee's Profile > release this email. To reactivate with the correct email, go to Staff > Disabled > the employee's Profile > Enable this employee."
3. **Using the same email for multiple employers** → "Employees can use Humanity for multiple employers by registering with different email addresses. Alternatively, they can employ an alias by appending '+1' before the '@' symbol in their email address. Example: `kyle+1@gmail.com` forwards emails to `kyle@gmail.com`. Some email providers may have different procedures."
4. **Viewing reports of a disabled employee** → "Are you interested in the scheduled hours report or worked hours report?"
5. **Editing shifts for a week** → "Could you specify the changes you need? I’ll be happy to assist further."
6. **Deleting availability** → "To delete your availability, go to the availability module, click on it, and select the option to delete your availability."
7. **Handling completely out-of-context questions** → "I’m unable to find information on that. Please reach out to your management for further assistance."
 further assistance."

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
        return "⚠️ Error: Document retrieval system is not available."

    # Retrieve the most relevant context from the document
    retrieved_docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    # ✅ Step 1: Ensure context exists (Preventing Hallucinations)
    if not context.strip():
        response = "I'm sorry, but I couldn't find relevant details for your question. Could you provide more specifics?"
        chat_memory.save_context(inputs={"input": question}, outputs={"output": response})
        return response

    # ✅ Step 2: Detect vague queries & force clarification
    vague_keywords = ["configure", "set up", "help with", "how do I manage", "how do I edit", "how do I adjust"]
    if any(keyword in question.lower() for keyword in vague_keywords):
        response = (
            "Could you clarify what exactly you want to configure? "
            "For example, are you looking to generate specific reports, export data, or change reporting settings?"
        )
        chat_memory.save_context(inputs={"input": question}, outputs={"output": response})
        return response

    # ✅ Step 3: If context is valid, proceed with answering the question
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

    port = int(os.getenv("PORT", 8000))  # Use Render's assigned PORT
    uvicorn.run(app, host="0.0.0.0", port=port)
