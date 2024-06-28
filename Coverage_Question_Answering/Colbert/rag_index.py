from ragatouille import RAGPretrainedModel
from ragatouille.utils import get_wikipedia_page

RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
# Here the data will go to index
my_documents =  [""]
index_path = RAG.index(index_name="my_index", collection=my_documents, max_document_length=256)