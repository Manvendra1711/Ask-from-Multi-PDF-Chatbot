import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from htmlTemplates import css, bot_template, user_template
import google.generativeai as genai
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
from pydantic import Field
import pymongo
from datetime import datetime
import hashlib
import uuid
import re
import shutil  # Added for directory cleanup

# MongoDB Configuration
class MongoDBHandler:
    def __init__(self):
        self.mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        self.db_name = os.getenv("MONGODB_DB_NAME", "pdf_chat_app")
        self.client = None
        self.db = None
        self.connect()
    
    def is_connected(self):
        """Check if database is connected"""
        try:
            return self.client is not None and self.db is not None
        except:
            return False
    
    def connect(self):
        try:
            self.client = pymongo.MongoClient(self.mongo_uri)
            self.db = self.client[self.db_name]
            # Test connection
            self.client.admin.command('ping')
            print("MongoDB connected successfully")
        except Exception as e:
            print(f"MongoDB connection failed: {str(e)}")
            st.warning(f"MongoDB connection failed: {str(e)}. App will continue without logging.")
            self.client = None
            self.db = None
    
    def validate_email(self, email):
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def hash_password(self, password):
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_user(self, email, password):
        """Create a new user account"""
        if not self.is_connected():
            return None, "Database not connected"
        
        # Validate email format
        if not self.validate_email(email):
            return None, "Please enter a valid email address"
        
        try:
            # Check if email already exists
            existing_user = self.db.users.find_one({"email": email})
            if existing_user:
                return None, "Email already exists"
            
            # Create unique user ID
            user_id = str(uuid.uuid4())
            
            user_data = {
                "user_id": user_id,
                "email": email,
                "password_hash": self.hash_password(password),
                "created_at": datetime.now(),
                "total_sessions": 0,
                "status": "active"
            }
            
            result = self.db.users.insert_one(user_data)
            return user_id, "User created successfully"
            
        except Exception as e:
            return None, f"Error creating user: {str(e)}"
    
    def authenticate_user(self, email, password):
        """Authenticate user login"""
        if not self.is_connected():
            return None, "Database not connected"
        
        # Validate email format
        if not self.validate_email(email):
            return None, "Please enter a valid email address"
        
        try:
            user = self.db.users.find_one({"email": email})
            if not user:
                return None, "Email not found"
            
            if user["password_hash"] == self.hash_password(password):
                return user["user_id"], "Login successful"
            else:
                return None, "Incorrect password"
                
        except Exception as e:
            return None, f"Error during authentication: {str(e)}"
    
    def create_user_session(self, user_id, email):
        """Create a new user session"""
        if not self.is_connected():
            return None
        
        session_data = {
            "session_id": str(uuid.uuid4()),
            "user_id": user_id,
            "email": email,
            "created_at": datetime.now(),
            "last_active": datetime.now(),
            "total_questions": 0,
            "total_pdfs_processed": 0,
            "status": "active"
        }
        
        try:
            result = self.db.user_sessions.insert_one(session_data)
            
            # Update user's total sessions count
            self.db.users.update_one(
                {"user_id": user_id},
                {"$inc": {"total_sessions": 1}}
            )
            
            return session_data["session_id"]
        except Exception as e:
            st.error(f"Error creating user session: {str(e)}")
            return None
    
    def log_conversation(self, session_id, user_id, email, question, answer, pdf_names=None, response_time=None):
        """Log individual conversation"""
        if not self.is_connected():
            return False
        
        conversation_data = {
            "session_id": session_id,
            "user_id": user_id,
            "email": email,
            "question": question,
            "answer": answer,
            "pdf_names": pdf_names or [],
            "timestamp": datetime.now(),
            "response_time_seconds": response_time,
            "question_length": len(question),
            "answer_length": len(answer)
        }
        
        try:
            # Insert conversation
            self.db.conversations.insert_one(conversation_data)
            
            # Update user session stats
            self.db.user_sessions.update_one(
                {"session_id": session_id},
                {
                    "$inc": {"total_questions": 1},
                    "$set": {"last_active": datetime.now()}
                }
            )
            return True
        except Exception as e:
            st.error(f"Error logging conversation: {str(e)}")
            return False
    
    def log_pdf_processing(self, session_id, user_id, email, pdf_names, total_chunks, processing_time):
        """Log PDF processing activity"""
        if not self.is_connected():
            return False
        
        pdf_data = {
            "session_id": session_id,
            "user_id": user_id,
            "email": email,
            "pdf_names": pdf_names,
            "total_pdfs": len(pdf_names),
            "total_chunks_created": total_chunks,
            "processing_time_seconds": processing_time,
            "timestamp": datetime.now()
        }
        
        try:
            # Insert PDF processing log
            self.db.pdf_processing.insert_one(pdf_data)
            
            # Update user session
            self.db.user_sessions.update_one(
                {"session_id": session_id},
                {
                    "$inc": {"total_pdfs_processed": len(pdf_names)},
                    "$set": {"last_active": datetime.now()}
                }
            )
            return True
        except Exception as e:
            st.error(f"Error logging PDF processing: {str(e)}")
            return False
    
    def get_user_stats(self, user_id):
        """Get user statistics"""
        if not self.is_connected():
            return None
        
        try:
            # Get user info
            user = self.db.users.find_one({"user_id": user_id})
            if not user:
                return None
            
            # Get user sessions
            sessions = list(self.db.user_sessions.find({"user_id": user_id}))
            
            # Get conversation count
            total_conversations = self.db.conversations.count_documents({"user_id": user_id})
            
            # Get PDF processing count
            total_pdfs = self.db.pdf_processing.count_documents({"user_id": user_id})
            
            return {
                "email": user["email"],
                "user_id": user_id,
                "total_sessions": len(sessions),
                "total_conversations": total_conversations,
                "total_pdfs_processed": total_pdfs,
                "member_since": user["created_at"].strftime("%Y-%m-%d"),
                "sessions": sessions
            }
        except Exception as e:
            st.error(f"Error getting user stats: {str(e)}")
            return None

class GeminiLLM(LLM):
    """Enhanced Custom LangChain LLM wrapper for Google Gemini with better accuracy"""
    
    model_name: str = Field(default="gemini-1.5-flash")
    gemini_model: Any = Field(default=None)
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash", **kwargs):
        super().__init__(**kwargs)
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.gemini_model = genai.GenerativeModel(model_name)
    
    @property
    def _llm_type(self) -> str:
        return "gemini"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            # Enhanced generation config for better accuracy
            generation_config = {
                "temperature": 0.3,  # Lower temperature for more focused answers
                "top_p": 0.9,        # Slightly higher for better context understanding
                "top_k": 40,
                "max_output_tokens": 4096,  # Increased for more detailed answers
            }
            
            # Enhanced prompt with better instructions
            enhanced_prompt = f"""You are an expert document analyst. Answer the question based ONLY on the provided context from the documents. 

Important guidelines:
1. If the information is not in the context, clearly state "I don't have enough information in the provided documents to answer this question."
2. Be specific and detailed when the information is available
3. Quote relevant parts from the documents when possible
4. If the question asks for multiple points, provide a structured answer
5. Always be accurate and don't make assumptions beyond what's stated in the documents

Context and Question:
{prompt}

Answer:"""
            
            response = self.gemini_model.generate_content(
                enhanced_prompt,
                generation_config=generation_config
            )
            
            # Check if response has text
            if hasattr(response, 'text') and response.text:
                return response.text
            elif hasattr(response, 'candidates') and response.candidates:
                # Try to get text from candidates
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and candidate.content.parts:
                        return candidate.content.parts[0].text
            
            return "No response generated from the model"
            
        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg or "not found" in error_msg.lower():
                # Try with a different model
                try:
                    backup_model = genai.GenerativeModel("gemini-1.5-pro")
                    backup_response = backup_model.generate_content(enhanced_prompt)
                    if hasattr(backup_response, 'text') and backup_response.text:
                        return backup_response.text
                except:
                    pass
                return f"Model not available. Please check available models in your region. Error: {error_msg}"
            return f"Error generating response: {error_msg}"
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": self.model_name}

def clear_vector_store_directories():
    """Clear existing vector store directories to ensure fresh start"""
    directories_to_clear = ["./chroma_db", "./chroma_db_fallback"]
    
    for directory in directories_to_clear:
        try:
            if os.path.exists(directory):
                shutil.rmtree(directory)
                print(f"Cleared directory: {directory}")
        except Exception as e:
            print(f"Could not clear directory {directory}: {str(e)}")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:  # Check if text is not empty
                    # Add page context for better retrieval
                    text += f"\n--- Page {page_num + 1} of {pdf.name} ---\n"
                    text += page_text + "\n"
        except Exception as e:
            st.error(f"Error reading PDF {pdf.name}: {str(e)}")
            continue
    
    if not text or len(text.strip()) == 0:
        st.error("No text could be extracted from the uploaded PDFs!")
        return None
    
    return text

def get_text_chunks(text):
    """Improved text chunking for better context preservation"""
    if not text or len(text.strip()) == 0:
        st.error("No text available to create chunks!")
        return []
    
    # Using RecursiveCharacterTextSplitter for better semantic chunking
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],  # Better separation hierarchy
        chunk_size=1500,      # Larger chunks for better context
        chunk_overlap=300,    # More overlap for context continuity
        length_function=len,
        is_separator_regex=False
    )
    chunks = text_splitter.split_text(text)
    
    # Filter out empty chunks and very short chunks
    chunks = [chunk.strip() for chunk in chunks if chunk.strip() and len(chunk.strip()) > 50]
    
    if not chunks:
        st.error("No valid text chunks could be created!")
        return []
    
    st.info(f"Created {len(chunks)} optimized text chunks from your documents.")
    return chunks

def get_vectorstore(text_chunks):
    """Enhanced vector store with better embedding model - Always creates fresh vector store"""
    if not text_chunks or len(text_chunks) == 0:
        st.error("No text chunks available to create vector store!")
        return None
    
    # Clear existing vector stores first
    clear_vector_store_directories()
    
    try:
        # Using a better embedding model for improved semantic understanding
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",  # Better model
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Test embedding with first chunk to ensure it works
        test_embedding = embeddings.embed_query(text_chunks[0])
        if not test_embedding:
            st.error("Failed to create embeddings!")
            return None
        
        # Create fresh vector store with unique directory name based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        persist_dir = f"./chroma_db_{timestamp}"
        
        vectorstore = Chroma.from_texts(
            texts=text_chunks, 
            embedding=embeddings,
            persist_directory=persist_dir
        )
        
        st.success(f"Fresh vector store created successfully with {len(text_chunks)} documents!")
        return vectorstore
        
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        # Fallback to simpler model if the enhanced one fails
        try:
            st.info("Trying with fallback embedding model...")
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Create fresh fallback vector store
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fallback_persist_dir = f"./chroma_db_fallback_{timestamp}"
            
            vectorstore = Chroma.from_texts(
                texts=text_chunks, 
                embedding=embeddings,
                persist_directory=fallback_persist_dir
            )
            
            st.success(f"Fallback vector store created with {len(text_chunks)} documents!")
            return vectorstore
        except Exception as fallback_error:
            st.error(f"Both primary and fallback embedding models failed: {str(fallback_error)}")
            return None

def get_conversation_chain(vectorstore):
    """Enhanced conversation chain with better retrieval and prompting"""
    # Using Gemini LLM
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        st.error("GEMINI_API_KEY not found in environment variables!")
        return None
    
    try:
        llm = GeminiLLM(api_key=gemini_api_key)
        
        # Enhanced custom prompt template for better accuracy
        custom_prompt = PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template="""You are a helpful AI assistant specialized in analyzing documents. Use the following context from the documents to answer the question accurately.

IMPORTANT INSTRUCTIONS:
- Answer ONLY based on the provided context
- If information is not in the context, say "I don't have this information in the provided documents"
- Be specific and detailed when information is available
- Quote relevant parts from the documents when helpful
- Maintain conversation context from chat history when relevant

Previous conversation:
{chat_history}

Document context:
{context}

Question: {question}

Detailed Answer:"""
        )
        
        # Create fresh memory for new PDFs
        memory = ConversationBufferMemory(
            memory_key='chat_history', 
            return_messages=True,
            output_key='answer'  # Specify output key for better memory handling
        )
        
        # Enhanced retriever with better search parameters
        retriever = vectorstore.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance for diverse results
            search_kwargs={
                "k": 8,           # Retrieve more documents for better context
                "lambda_mult": 0.7  # Balance between relevance and diversity
            }
        )
        
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": custom_prompt},
            return_source_documents=True,  # Return source docs for verification
            verbose=False
        )
        return conversation_chain
    except Exception as e:
        st.error(f"Error creating conversation chain: {str(e)}")
        return None

def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.error("Please upload and process PDFs first!")
        return
    
    try:
        start_time = datetime.now()
        response = st.session_state.conversation({
            'question': user_question
        })
        end_time = datetime.now()
        response_time = (end_time - start_time).total_seconds()
        
        st.session_state.chat_history = response['chat_history']
        
        # Get the latest answer
        latest_answer = response.get('answer', 'No response generated')
        
        # Show source documents if available (for transparency)
        if 'source_documents' in response and response['source_documents']:
            with st.expander("📚 Source References", expanded=False):
                for i, doc in enumerate(response['source_documents'][:3]):  # Show top 3 sources
                    st.write(f"**Source {i+1}:** {doc.page_content[:200]}...")
        
        # Log conversation to MongoDB
        if st.session_state.mongo_handler is not None and st.session_state.session_id is not None:
            st.session_state.mongo_handler.log_conversation(
                session_id=st.session_state.session_id,
                user_id=st.session_state.user_id,
                email=st.session_state.email,
                question=user_question,
                answer=latest_answer,
                pdf_names=st.session_state.get('current_pdf_names', []),
                response_time=response_time
            )

        # Display chat history in reverse order (latest first)
        if st.session_state.chat_history:
            # Create pairs of question-answer
            chat_pairs = []
            for i in range(0, len(st.session_state.chat_history), 2):
                if i + 1 < len(st.session_state.chat_history):
                    question = st.session_state.chat_history[i].content
                    answer = st.session_state.chat_history[i + 1].content
                    chat_pairs.append((question, answer))
            
            # Display pairs in reverse order (latest first) with numbered labels
            for idx, (question, answer) in enumerate(reversed(chat_pairs)):
                question_number = len(chat_pairs) - idx
                st.write(user_template.replace("{{LABEL}}", f"Ques {question_number}").replace("{{MSG}}", question), unsafe_allow_html=True)
                st.write(bot_template.replace("{{LABEL}}", "Ans").replace("{{MSG}}", answer), unsafe_allow_html=True)
                
    except Exception as e:
        st.error(f"Error processing your question: {str(e)}")
        st.info("Please try rephrasing your question or check if your documents were processed correctly.")

def user_authentication_section():
    """User authentication section"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    if not st.session_state.logged_in:
        # Show login/register without toggle for non-logged-in users
        st.sidebar.subheader("🔐 User Authentication")
        
        # Login/Register tabs
        login_tab, register_tab = st.sidebar.tabs(["Login", "Register"])
        
        with login_tab:
            with st.form("login_form"):
                email = st.text_input("Email", key="login_email", placeholder="Enter your email address")
                password = st.text_input("Password", type="password", key="login_password")
                submit_button = st.form_submit_button("Login")
                
                if submit_button:
                    if not email or not password:
                        st.error("Please enter both email and password")
                    else:
                        user_id, message = st.session_state.mongo_handler.authenticate_user(email, password)
                        if user_id:
                            st.session_state.user_id = user_id
                            st.session_state.email = email
                            st.session_state.logged_in = True
                            
                            # Create user session in MongoDB
                            session_id = st.session_state.mongo_handler.create_user_session(user_id, email)
                            st.session_state.session_id = session_id
                            
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
        
        with register_tab:
            with st.form("register_form"):
                new_email = st.text_input("Email Address", key="register_email", placeholder="Enter your email address")
                new_password = st.text_input("Choose Password", type="password", key="register_password")
                confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
                register_button = st.form_submit_button("Register")
                
                if register_button:
                    if not new_email or not new_password or not confirm_password:
                        st.error("Please fill all fields")
                    elif new_password != confirm_password:
                        st.error("Passwords do not match")
                    elif len(new_password) < 6:
                        st.error("Password must be at least 6 characters long")
                    else:
                        user_id, message = st.session_state.mongo_handler.create_user(new_email, new_password)
                        if user_id:
                            st.success(message + " Please login now.")
                        else:
                            st.error(message)
    else:
        # Show user info in toggle after login
        with st.sidebar.expander("🤖 User Account", expanded=False):
            st.success("✅ Logged In")
            st.write(f"**Email:** {st.session_state.email}")
            if st.button("👾 Logout", key="logout_btn"):
                # Clear session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

def show_pdf_upload_section():
    """PDF upload section with enhanced processing"""
    st.sidebar.subheader("📄 Your Documents")
    pdf_docs = st.sidebar.file_uploader(
        "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
    
    if st.sidebar.button("🔄 Process"):
        if not pdf_docs:
            st.sidebar.error("Please upload at least one PDF file!")
            return
        
        start_time = datetime.now()
        with st.spinner("Processing PDFs with enhanced accuracy..."):
            try:
                # Clear previous conversation and chat history when processing new PDFs
                st.session_state.conversation = None
                st.session_state.chat_history = None
                
                # Store PDF names
                pdf_names = [pdf.name for pdf in pdf_docs]
                st.session_state.current_pdf_names = pdf_names
                
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                
                if not raw_text:
                    st.error("Could not extract text from PDFs. Please check if PDFs contain readable text.")
                    return

                # get the text chunks with improved algorithm
                text_chunks = get_text_chunks(raw_text)
                
                if not text_chunks:
                    st.error("Could not create text chunks. Please try with different PDFs.")
                    return

                # create vector store with enhanced embeddings (this now clears old data)
                vectorstore = get_vectorstore(text_chunks)
                
                if not vectorstore:
                    st.error("Could not create vector store. Please try again.")
                    return

                # create conversation chain with better prompting
                conversation_chain = get_conversation_chain(vectorstore)
                
                if conversation_chain:
                    st.session_state.conversation = conversation_chain
                    
                    # Log PDF processing to MongoDB
                    end_time = datetime.now()
                    processing_time = (end_time - start_time).total_seconds()
                    
                    if st.session_state.mongo_handler is not None and st.session_state.get('session_id') is not None:
                        st.session_state.mongo_handler.log_pdf_processing(
                            session_id=st.session_state.session_id,
                            user_id=st.session_state.user_id,
                            email=st.session_state.email,
                            pdf_names=pdf_names,
                            total_chunks=len(text_chunks),
                            processing_time=processing_time
                        )
                    
                    st.success(f"✅ Successfully processed {len(pdf_names)} PDFs! You can now ask questions.")
                    
                else:
                    st.error("Failed to create conversation chain!")
                    
            except Exception as e:
                st.error(f"Error processing PDFs: {str(e)}")
                st.info("Please try uploading different PDF files or check if they contain readable text.")

def show_user_stats():
    """Show user stats section with toggle"""
    with st.sidebar.expander("📊 Your Stats", expanded=False):
        if st.session_state.mongo_handler is not None and st.session_state.get('logged_in', False):
            stats = st.session_state.mongo_handler.get_user_stats(st.session_state.user_id)
            if stats:
                st.write(f"**Member Since:** {stats['member_since']}")
                st.write(f"**Total Sessions:** {stats['total_sessions']}")
                st.write(f"**Total Questions:** {stats['total_conversations']}")
                st.write(f"**Total PDFs Processed:** {stats['total_pdfs_processed']}")
        else:
            st.write("Please login to view your stats")

def main():
    load_dotenv()
    st.set_page_config(page_title="Enhanced PDF Chat - Better Accuracy",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Initialize MongoDB handler
    if "mongo_handler" not in st.session_state:
        st.session_state.mongo_handler = MongoDBHandler()

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "current_pdf_names" not in st.session_state:
        st.session_state.current_pdf_names = []

    # User authentication section (no toggle before login, toggle after login)
    user_authentication_section()
    
    # Only show main app sections if logged in
    if st.session_state.get('logged_in', False):
        # Show PDF upload section
        show_pdf_upload_section()
        
        # Show user stats (toggle)
        show_user_stats()
        
        # Main chat interface
        st.header("Chat with PDFs with Better Accuracy 📘:")
        
        # Show current PDFs info
        if st.session_state.current_pdf_names:
            with st.expander("📄 Currently Loaded PDFs", expanded=False):
                for i, pdf_name in enumerate(st.session_state.current_pdf_names, 1):
                    st.write(f"{i}. {pdf_name}")
        
        user_question = st.text_input("Ensure your questions are clear and free from spelling errors, To get the most appropriate answer:")
        if user_question:
            handle_userinput(user_question)

    else:
        st.header("🐾 Welcome to Ask from Multi PDF-Chatbot")
        st.write("Please login or register from the sidebar to start using the application.")

if __name__ == '__main__':
    main()