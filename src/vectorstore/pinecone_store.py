import chromadb
class ChromaDB:
    def __init__(self , file_path , collection_name:'my_docs'):

        self.client = ChromaDB.client()
        self.collection = self.client.get_or_create_collection(name=collection_name)

        





    
