import pandas as pd
from langchain_core.documents import Document

class DataConverter:
    #file path
    def __init__(self,file_path:str):
        self.file_path = file_path

#convert csv into dataframe because llm cant understand csv 
    def convert(self):
        df = pd.read_csv(self.file_path)[["product_title","review"]]   

        docs = [
            Document(page_content=row['review'] , metadata = {"product_name" : row["product_title"]})
            for _, row in df.iterrows()
        ]

        return docs

#in this we train owr model with product title and the review of the  product