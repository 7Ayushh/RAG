
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma  # Updated import
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from vec_store import vectordb

os.environ["GOOGLE_API_KEY"] = "AIzaSyA0ucMJyyWqa1GHiiVsUHaqGpP4REiq0IU"


template = """
You are an intelligent Question-Answer bot. Your task is to respond to questions by using the given context. Each answer should follow the structured format below, including the answer and the source of information.

### Response Format:
1. **Answer**: Provide a clear and complete answer to the question based only on the provided context.
2. **Source**: Explicitly mention the source of the answer in parentheses at the end. For example, 'Source: Section 2.5.1, Minor Program.'

If the answer is not in the context, say: "I don’t know based on the provided information." Do not attempt to make up an answer or provide false information. Use only the context to respond.

### Example:
**Answer**: IIT Bombay follows a 10-point grading system where the grades range from AP (Exceptional Performance) to FR (Fail).  
**Source**: Section 6.5, Grading System.

Now, use the context below to answer the question in the format provided.

Context:
{context}

Question: {input}
Answer with Source:
"""

# Now you can use this template with ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        ("human", "{input}")
    ]
)

questions = [
    "What is the grading system at IIT Bombay?",
    "What does the 'AP' grade signify at IIT Bombay?",
    "What is the significance of the 'PP' and 'NP' grades?",
    "Is attendance mandatory for all courses?",
    "What are the core academic phases of IIT Bombay's undergraduate programs?"
    "how is the end semester reev"
]

# Reference answers (corresponding to each question)
reference_answers = [
    "IIT Bombay follows a 10-point grading system where the grades range from 'AP' (Exceptional Performance) to 'FR' (Fail). Source: Grading System section.",
    "The 'AP' grade signifies exceptional performance and is awarded only in courses where the number of registered students is more than 50. Source: Grading System section.",
    "The 'PP' (Pass) and 'NP' (Not Pass) grades are used for non-credit courses such as NCC, NSO, and NSS. Source: Grading System section.",
    "Yes, IIT Bombay expects 100percent attendance from its students. A 'DX' grade is given if attendance falls below 80%. Source: Attendance section.",
    "The undergraduate program consists of three phases: intense study of sciences and humanities, study of engineering sciences, and specialized subjects in chosen areas. Source: Introduction section."
]

k=10
score = 0.9
temp = 0.1
max_toke = 200

llm = GoogleGenerativeAI(model="gemini-pro", max_tokens=max_toke, temperature=temp)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(vectordb.as_retriever(), question_answer_chain)

for i, question in enumerate(questions):
    # Invoke rag_chain for the question
    result = rag_chain.invoke({"input": question})
    
    # Print the question and both responses
    print(f"Question {i+1}: {question}")
    print(f"Our Bot Answer: {result['answer']}")
    print(f"GPT Answer: {reference_answers[i]}")
    
    # Output separator
    print("\n" + "-"*50 + "\n")