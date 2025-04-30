# import streamlit as st
# from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader  # Updated import
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import GoogleGenerativeAI
# from langchain.vectorstores import Chroma
# from langchain.prompts import PromptTemplate
# import streamlit as st
# import param



def load_db(file, chain_type, k):
    loader = PyPDFLoader(file)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key="AIzaSyC5sCAPSQA4DfGsH-2tKhRW0VMnIGINEkU")
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})

    qa = ConversationalRetrievalChain.from_llm(
        llm=GoogleGenerativeAI(model="gemini-pro", google_api_key="AIzaSyC5sCAPSQA4DfGsH-2tKhRW0VMnIGINEkU"),
        chain_type=chain_type, 
        retriever=retriever, 
        return_source_documents=True,
        return_generated_question=True,
    )
    return qa
    

# # Sample function to simulate database querying and response generation

# class cbfs(param.Parameterized):
#     chat_history = param.List([])
#     answer = param.String("")
#     db_query  = param.String("")
#     db_response = param.List([])
    
#     def __init__(self,  **params):
#         super(cbfs, self).__init__( **params)
#         self.loaded_file = "ugrulebook.pdf"
#         self.qa = load_db(self.loaded_file, "stuff", 4)
    
#     def call_load_db(self, file):
#         if not file:
#             return "No file loaded."
#         else:
#             self.loaded_file = file.name
#             self.qa = load_db(file.name, "stuff", 4)
#         self.clr_history()
#         return f"Loaded File: {self.loaded_file}"

#     def convchain(self, query):
#         if not query:
#             return None
#         result = self.qa({"question": query, "chat_history": self.chat_history})
#         self.chat_history.extend([(query, result["answer"])])
#         self.db_query = result["generated_question"]
#         self.db_response = result["source_documents"]
#         self.answer = result['answer'] 
#         return result['answer']

#     def get_lquest(self):
#         if not self.db_query:
#             return "No DB query so far"
#         return f"DB query: {self.db_query}"

#     def get_sources(self):
#         if not self.db_response:
#             return "No DB response"
#         return f"Result of DB lookup: {', '.join(self.db_response)}"

#     def get_chats(self):
#         if not self.chat_history:
#             return "No Chat History Yet"
#         return "\n".join([f"User: {q}, Bot: {a}" for q, a in self.chat_history])

#     def clr_history(self):
#         self.chat_history = []


# # Create an instance of cbfs class
# cb = cbfs()

# st.title('ChatWithYourData_Bot')

# # File upload
# file = st.file_uploader("Upload a PDF", type="pdf")
# if file:
#     st.success(cb.call_load_db(file))

# # Chat interface
# query = st.text_input("Ask something to the chatbot")
# if st.button('Send'):
#     if query:
#         answer = cb.convchain(query)
#         st.write(f"User: {query}")
#         st.write(f"ChatBot: {answer}")

# # Chat history
# if st.button('Clear Chat History'):
#     cb.clr_history()

# st.write("### Chat History")
# st.text(cb.get_chats())

# # Database query and response
# st.write("### Last DB Query")
# st.text(cb.get_lquest())

# st.write("### Source Documents")
# st.text(cb.get_sources())


import streamlit as st
from langchain_community.document_loaders import PyPDFLoader  # Updated import
from langchain_community.vectorstores import DocArrayInMemorySearch, Chroma  # Updated imports

class ChatBot:
    def __init__(self):
        self.db_response = []

    def load_data(self, pdf_file):
        loader = PyPDFLoader(pdf_file)
        documents = loader.load()
        self.db_response = documents

    def get_sources(self):
        # Extracting 'page_content' or 'metadata' from Document objects
        source_texts = [doc.page_content for doc in self.db_response]  # Assuming 'page_content' holds the text
        return f"Result of DB lookup: {', '.join(source_texts)}"

    def process_query(self, query):
        # Processing query and getting response from a vectorstore (for example purposes)
        result = self.vector_search(query)
        self.db_response = result["source_documents"]

    def vector_search(self, query):
        # Dummy implementation for vector search
        return {"source_documents": self.db_response}


# Streamlit app code
cb = ChatBot()

# Load a PDF file (Example usage)
pdf_file = "example.pdf"  # Replace with your file path
cb.load_data(pdf_file)

# Display result
st.text(cb.get_sources())


