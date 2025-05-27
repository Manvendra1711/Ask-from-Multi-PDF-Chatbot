# Ask-from-Multi-PDF-Chatbot
👾 AI-powered PDF chat app using Google Gemini &amp; LangChain. Upload PDFs, ask questions, get intelligent answers with source references. Built with Streamlit &amp; MongoDB.

# 📚 PDF Chat Application

An intelligent document analysis system that enables users to upload multiple PDF documents and engage in natural language conversations about their content. Built with Streamlit, LangChain, and Google's Gemini AI for enhanced accuracy and user experience.

## ✨ Features

- **Multi-PDF Processing**
- **Intelligent Conversations**: Ask questions about your documents in natural language
- **User Authentication**: Secure login/registration system with MongoDB integration
- **Session Management**: Track user sessions, questions, and document processing history
- **Responsive UI**: Clean, modern interface with chat history management

## 🔧 How It Works

1. **Document Processing**: PDFs are parsed and text is extracted using PyPDF2
2. **Text Chunking**: Content is intelligently split into overlapping chunks for better context preservation
3. **Vector Embeddings**: Text chunks are converted to embeddings using HuggingFace transformers
4. **Vector Storage**: Embeddings are stored in ChromaDB for efficient similarity search
5. **Question Processing**: User questions are matched against relevant document chunks
6. **AI Response**: Google Gemini generates contextual answers based on retrieved content
7. **Conversation Memory**: Chat history is maintained for follow-up questions

## 🛠️ Technologies Used

- **Frontend**: Streamlit
- **AI/ML**: Google Gemini AI, LangChain, HuggingFace Transformers
- **Database**: MongoDB (user management), ChromaDB (vector storage)
- **PDF Processing**: PyPDF2
- **Text Processing**: RecursiveCharacterTextSplitter
- **Embeddings**: sentence-transformers

## 📋 Prerequisites

- Python 3.8 or higher
- MongoDB instance (local or cloud)
- Google AI API key (for Gemini)

## 🚀 Installation

1. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```env
   GEMINI_API_KEY=your_google_gemini_api_key
   MONGODB_URI=mongodb://localhost:29573/
   MONGODB_DB_NAME=pdf_chat_app
   ```

4. **Set up MongoDB**
   - Install MongoDB locally or use MongoDB Atlas
   - Update the `MONGODB_URI` in your `.env` file
   - The application will automatically create required collections

## 📖 Usage

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Access the application**
   - Open your browser and navigate to `http://localhost:8501`

3. **User Registration/Login**
   - Create a new account or login with existing credentials
   - User authentication is required to access the chat features

4. **Upload PDFs**
   - Use the sidebar to upload one or multiple PDF files
   - Click "Process" to analyze the documents
   - Wait for the processing to complete

5. **Start Chatting**
   - Ask questions about your uploaded documents

## 🔑 API Keys Setup

### Google Gemini API Key
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add it to your `.env` file as `GEMINI_API_KEY`

### MongoDB Setup
- **Local MongoDB**: Use `mongodb://localhost:29582/`
- **Database Name**: Default is `pdf_chat_app` (configurable)

## 📊 Project Structure

```
pdf-chat-app/
├── app.py              # Main application file
├── htmlTemplates.py    # HTML/CSS templates for chat UI
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (create this)
├── README.md          # Project documentation
└── chroma_db/         # Vector database (auto-created)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Happy chatting with your PDFs! 🚀**
