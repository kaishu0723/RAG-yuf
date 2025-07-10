import pandas as pd
from langchain.docstore.document import Document

data_path="./data/spot-data.csv"

df=pd.read_csv(data_path,"utf-8")

documents=[]
for _,row in df.iterrows():
    page_content=f"スポット名{row["Title"]}\n説明{row["spot_editor"]}"
    metadata={
        "source":data_path,
        "title":row["Title"]
    }
    
    doc=Document(page_content,metadata)
    documents.append(doc)