import os
import base64
import unicodedata
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain


def clean_text(text):
    return unicodedata.normalize('NFKD', ''.join(char for char in text if not 0xD800 <= ord(char) <= 0xDFFF))


def get_pdf_text(upload_pdfs):
    text = []
    temp_file_path = os.path.join(os.getcwd(), upload_pdfs.name)
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(upload_pdfs.read())

    try:
        pdf_reader = PyPDFLoader(temp_file_path)
        pages = pdf_reader.load()
        cleaned_pages = []
        for page in pages:
            page.page_content = clean_text(page.page_content)
            cleaned_pages.append(page)

        text.extend(cleaned_pages)

    except Exception as e:
        print(f"Error processing PDF {upload_pdfs.name}: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    return text

def display_pdf_in_sidebar(pdf_uploader):
    if pdf_uploader is not None:
        try:
            # Use the file's binary content for the viewer
            pdf_data = pdf_uploader.getvalue()

            # Encode PDF to base64 for rendering
            base64_pdf = base64.b64encode(pdf_data).decode("utf-8")

            # Display PDF in the sidebar
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px" type="application/pdf"></iframe>'
            st.sidebar.markdown(pdf_display, unsafe_allow_html=True)

        except Exception as e:
            st.sidebar.error(f"An error occurred while displaying the PDF: {e}")


def splitting_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                   chunk_overlap=100,
                                                   separators='\n',
                                                   length_function=len)

    chunks = text_splitter.split_documents(text)
    return chunks


def store_chunks_vectorDB(chunks):
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectore_DB = FAISS.from_documents(documents=chunks, embedding=embedding)
    return vectore_DB



def main():
    load_dotenv()
    st.set_page_config(page_title='Chat With PDFs', page_icon='📚', layout='wide')
    st.title('Chat With PDFs📚')


    st.sidebar.subheader('Upload Section.')
    pdf_uploader = st.sidebar.file_uploader('Upload Your PDF: ', type='pdf')
    if pdf_uploader:
        display_pdf_in_sidebar(pdf_uploader)

        raw_text = get_pdf_text(pdf_uploader)

        # Splitting Data Into Chunks
        chunks = splitting_text(raw_text)

        # Save Chunks into vector-Database
        vectors = store_chunks_vectorDB(chunks=chunks)

        user_query = st.chat_input('Enter the Question: ')
        if user_query:
            docs = vectors.similarity_search(query=user_query, k=4)

            llm = ChatGroq(model='llama3-70b-8192')
            chain = load_qa_chain(llm=llm, chain_type='stuff')
            response = chain.run(input_documents=docs, question=user_query)

            with st.chat_message('user'):
                st.write(user_query)

            with st.chat_message('assistant'):
                st.write(response)


if __name__ == '__main__':
    main()