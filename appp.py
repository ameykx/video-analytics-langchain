from google.cloud import aiplatform
# from flask import Flask, render_template, request,session
import os
import vertexai
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_vertexai import VertexAI
from langchain_google_vertexai import VertexAIEmbeddings


# PROJECT_ID = os.environ.get('GCP_PROJECT') #Your Google Cloud Project ID
# LOCATION = os.environ.get('GCP_REGION')   #Your Google Cloud Project Region

vertexai.init(project=PROJECT_ID, location=LOCATION)

llm = VertexAI(
model_name="text-bison@002",
max_output_tokens=256,
temperature=0.1,
top_p=0.8,
top_k=40,
verbose=True,
)


# Embedding
EMBEDDING_QPM = 100
EMBEDDING_NUM_BATCH =5
embeddings = VertexAIEmbeddings(
    model_name="textembedding-gecko@003",
    requests_per_minute=EMBEDDING_QPM,
    num_instances_per_batch=EMBEDDING_NUM_BATCH,
)

user_input_link="https://youtu.be/rfscVS0vtbw?si=KNJ-VBDbPTZRthGx"
loader = YoutubeLoader.from_youtube_url(user_input_link, add_video_info=True)
result = loader.load()

print(result) #video transcript

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0) #chunking the transcripts
docs = text_splitter.split_documents(result)

db = Chroma.from_documents(docs, embeddings) #stores embedding in chromadb
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})

global qaa
qaa = RetrievalQA.from_chain_type( llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True) #chain object

def sm_ask(question, print_results=True):
  video_subset = qaa.invoke({"query": question})
  context = video_subset
  prompt = f"""
  Answer the following question in a detailed manner, using information from the text below. If the answer is not in the text,say I dont know and do not generate your own response.

  Question:
  {question}
  Text:
  {context}

  Question:
  {question}

  Answer:
  """
  parameters = {
  "temperature": 0.1,
  "max_output_tokens": 256,
  "top_p": 0.8,
  "top_k": 40
  }
  response = llm.invoke(prompt, **parameters)
  return response


user_input="list the data types in python"
content = sm_ask(user_input)
print(content)

    
