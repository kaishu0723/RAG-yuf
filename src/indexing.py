import pandas as pd
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
import pickle


data_path="./data/spot-data.csv"
df=pd.read_csv(data_path)


documents=[]
for _,row in df.iterrows():
    page_content=f"スポット名{row["Title"]}\n説明{row["spot_editor"]}"
    metadata={
        "source":data_path,
        "title":row["Title"]
    }
    
    doc=Document(page_content=page_content,metadata=metadata)
    documents.append(doc)


embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore=Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./vectorstore",
)

bm_ret=BM25Retriever.from_documents(documents)
with open("./bm25.pkl",'wb') as f:
    pickle.dump(bm_ret,f)