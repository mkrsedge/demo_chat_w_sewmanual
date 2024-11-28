from openai import OpenAI
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores.faiss import FAISS
from langchain.callbacks import get_openai_callback
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
import os
import pickle
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())

api_key = os.environ.get("OPEN_AI_KEY")

embeddings = OpenAIEmbeddings(
   model = "text-embedding-ada-002",
   openai_api_key = api_key,
   chunk_size = 1
)

llm = ChatOpenAI(
   model = "gpt-3.5-turbo",
   openai_api_key = api_key
)

def page1():
   
   st.title("Maker's Edge - Chat with Operator Manual")
   pdf = "data/mvp_pdf.pdf"

   st.write("Here you can chat with the manual!")
   query_text = st.text_input("Enter your question:", disabled=not pdf, key=0)


   if pdf is not None:
      pdf_reader = PdfReader(pdf)
      text = ""

      for page in pdf_reader.pages:
         text += page.extract_text()

      text_splitter = RecursiveCharacterTextSplitter(
         chunk_size = 2000,
         chunk_overlap = 200,
         length_function = len
      )

      chunks = text_splitter.split_text(text=text)

      #embeddings
      store_name = pdf[8:-4]
      if os.path.exists("vectorstore"):
         Vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

      else:
         Vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
         Vectorstore.save_local("vectorstore")


      #user query
      if query_text:
         
         docs = Vectorstore.similarity_search(query=query_text, k=3)
         chain = load_qa_chain(llm=llm, chain_type="stuff")

         with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=query_text)
            print(cb)

         st.write(response)

         

    


#siderbar navigation
sidebar_options = {
   "Customer Services - Chat with Sewing Machine Manual": page1
}


def main():
   
   st.sidebar.title("Maker's Edge OpenAI & LLM Demos")
   page = st.sidebar.radio("", list(sidebar_options.keys()))

   #Execute the selected page function
   sidebar_options[page]()

if __name__ == "__main__":
   main()