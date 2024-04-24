from haystack.components.writers import DocumentWriter
from haystack.components.converters import MarkdownToDocument, PyPDFToDocument, TextFileToDocument
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.routers import FileTypeRouter
from haystack.components.joiners import DocumentJoiner
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack_integrations.components.embedders.fastembed import FastembedDocumentEmbedder
from haystack.document_stores.types import DuplicatePolicy
from haystack import Pipeline
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.components.embedders.fastembed import FastembedTextEmbedder
# from haystack.components.generators import HuggingFaceTGIGenerator
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator
from haystack.components.builders.answer_builder import AnswerBuilder
import time

# file_path = '/home/kevin/ai/books/dnd/file/HoardDragonQueen_Supplement_PF_v0.3.pdf'
# pdf_converter = PyPDFToDocument()
# pdf_doc = pdf_converter.run(sources=[file_path])


# cleaner = DocumentCleaner()
# splitter = DocumentSplitter(split_by="word", split_length=50, split_overlap=10)
# splitted_docs = splitter.run(cleaner.run(pdf_doc["documents"])["documents"])

# print(len(splitted_docs["documents"]), 'chunks')
# document_embedder = FastembedDocumentEmbedder(model="BAAI/bge-small-en-v1.5", parallel = 0)
# document_embedder.warm_up()
# documents_with_embeddings = document_embedder.run(splitted_docs["documents"])
# document_store.write_documents(documents_with_embeddings.get("documents"), policy=DuplicatePolicy.OVERWRITE)


# constants
model_path = "/home/kevin/ai/models/q4_0-bling-sheared-llama-1.3b-0.1.gguf"
prompt_template = """<human>: Please read the following text: {% for doc in documents %}
{{doc.content}}{% endfor %}
Based on this text, please answer the question: 
{{question}}\n<bot>:"""


# inits 
document_store = QdrantDocumentStore(
    path="/home/kevin/ai/books/dnd/db",
    embedding_dim =384,
    recreate_index=False,
    return_embedding=True,
    wait_result_from_api=True,
)
generator = LlamaCppGenerator(
    model=model_path, 
    n_ctx=512, 
    n_batch=1,
    generation_kwargs={"max_tokens": 128,  "temperature" : 0.3, "stop" : ["</s>", "<|im_end|>", "<|endoftext|>"]},
)
generator.warm_up()
# define the prompt template

query_pipeline = Pipeline()
# FastembedTextEmbedder is used to embed the query
query_pipeline.add_component("text_embedder", FastembedTextEmbedder(model="BAAI/bge-small-en-v1.5", parallel = 0, prefix="query:"))
query_pipeline.add_component("retriever", QdrantEmbeddingRetriever(document_store=document_store,top_k=1))
query_pipeline.add_component("prompt_builder", PromptBuilder(template=prompt_template))
query_pipeline.add_component(instance=AnswerBuilder(), name="answer_builder")
query_pipeline.add_component("generator", generator)

# connect the components
query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
query_pipeline.connect("retriever.documents", "prompt_builder.documents")
query_pipeline.connect("prompt_builder", "generator")
query_pipeline.connect("generator.replies", "answer_builder.replies")
query_pipeline.connect("retriever.documents", "answer_builder.documents")
query_pipeline.connect("prompt_builder.prompt", "answer_builder.query")



def main():
    while True:
        question = input("Enter your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            print("Exiting the program.")
            break
        try:
            st = time.time()
            results = query_pipeline.run(
                {   "text_embedder": {"text": question},
                    "prompt_builder": {"question": question},
                    # "answer_builder": {"query": question},
                }
            )
            time_taken = time.time() - st
            print(f"Query done. Time taken: {time_taken:.2f} seconds")
            print(f"Question: {results['answer_builder']['answers'][0].query}")
            print(f"Answer: {results['answer_builder']['answers'][0].data}")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
