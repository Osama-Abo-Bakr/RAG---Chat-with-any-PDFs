# RAG: Chat with any PDFs 📚

A Streamlit-based application that leverages Retrieval-Augmented Generation (RAG) to allow users to interact with PDF documents. Users can upload PDFs, ask questions about their contents, and receive AI-generated responses based on the uploaded documents.

---

## Features
- **PDF Upload**: Upload and view PDFs directly in the app sidebar.
- **Text Cleaning**: Ensures text integrity by normalizing Unicode and removing invalid characters.
- **Text Splitting**: Splits large PDF content into manageable chunks for efficient processing.
- **Vector Database**: Creates and stores embeddings for document chunks using Google Generative AI Embeddings.
- **Question Answering**: Uses Groq’s `llama3-70b-8192` model to answer user queries based on the uploaded PDFs.

---

## How It Works
1. **PDF Upload**:
   - Users upload PDFs, which are rendered in the sidebar for easy viewing.
   - The text content of the PDF is extracted and cleaned.

2. **Text Processing**:
   - The extracted text is split into chunks using the `RecursiveCharacterTextSplitter`.

3. **Embedding and Storage**:
   - Each text chunk is embedded using the `GoogleGenerativeAIEmbeddings` model.
   - The embeddings are stored in a FAISS vector database.

4. **Question Answering**:
   - Users submit queries through a chat interface.
   - Relevant chunks are retrieved using similarity search.
   - Answers are generated using Groq’s `llama3-70b-8192` model.

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Osama-Abo-Bakr/RAG-Chat-with-any-PDFs.git
   cd RAG-Chat-with-any-PDFs
   ```

2. **Install Dependencies**:
   - Create a virtual environment:
     ```bash
     python -m venv venv
     source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
     ```
   - Install required packages:
     ```bash
     pip install -r requirements.txt
     ```

3. **Set Up Environment Variables**:
   - Create a `.env` file in the project directory and add your credentials:
     ```plaintext
     GOOGLE_API_KEY=your_google_api_key
     GROQ_API_KEY=your_groq_api_key
     ```

---

## Usage

1. **Run the Application**:
   ```bash
   streamlit run main.py
   ```

2. **Upload and Interact**:
   - Use the sidebar to upload your PDF.
   - Enter your questions in the chat input.
   - View the AI-generated answers in real time.

---

## Dependencies
- **[Streamlit](https://streamlit.io/)**: For building the interactive web application.
- **[LangChain](https://langchain.com/)**: For document loaders, text splitting, and chains.
- **[FAISS](https://faiss.ai/)**: For efficient similarity search.
- **[Google Generative AI](https://cloud.google.com/generative-ai/)**: For generating embeddings.
- **[Groq](https://groq.com/)**: For large language model-based question answering.

---

## Future Improvements
- Add support for multi-file uploads.
- Enhance UI/UX for better user experience.
- Integrate additional document formats (e.g., Word, TXT).
- Allow saving and exporting of chat interactions.

---

## Contributing
Contributions are welcome! Feel free to fork the repository and submit a pull request.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Author
**Osama Abo Bakr**  
GitHub: [Osama-Abo-Bakr](https://github.com/Osama-Abo-Bakr)

---

## Acknowledgments
- Special thanks to the developers of Streamlit, LangChain, FAISS, Google Generative AI, and Groq for their amazing tools!