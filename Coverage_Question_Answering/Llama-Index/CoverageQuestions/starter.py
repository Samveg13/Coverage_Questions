from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

PERSIST_DIR = "./storage"

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
index.storage_context.persist(persist_dir=PERSIST_DIR)
