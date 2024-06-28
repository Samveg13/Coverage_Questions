from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding

import os
# load documents
documents = SimpleDirectoryReader(input_files=["/Users/samveg.shah/Desktop/Llama-index/data/Your Obligations as an Insured Endorsement_NAS_NAC - SP 17 275 0219.pdf"]).load_data()

from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
)

embed_model = OpenAIEmbedding()
splitter = SemanticSplitterNodeParser(
    buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model
)

# also baseline splitter
base_splitter = SentenceSplitter(chunk_size=512)
nodes = splitter.get_nodes_from_documents(documents)
print(nodes[0].get_content())
print(nodes[1].get_content())

# print(nodes[2].get_content())
# print(nodes[3].get_content())
# print(nodes[4].get_content())
# print(nodes[5].get_content())
# print(nodes[6].get_content())
