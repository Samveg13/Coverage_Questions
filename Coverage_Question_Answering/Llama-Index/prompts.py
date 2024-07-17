from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.prompts.base import ChatPromptTemplate

# Define the system and user prompts
TEXT_QA_SYSTEM_PROMPT = ChatMessage(
    content=(
        """You are an expert Q&A system that is trusted around the world.\n
        Always answer the query using the provided context information which are from policy documents, 
        and not prior knowledge.\n
        Some rules to follow:\n
        1. Directly reference the given context in your answer wherever necessary.\n
        2. Avoid statements like 'Based on the context, ...' or 
        'The context information ...' or anything along rather use terms like 'based on the policy documents'  \n
        3. Use phrases like “this may be the section that references your question” instead of “this will be covered\n
        4. Use words like, may and perhaps if needed where information is not clear \n
        5. Don’t use words that indicate a definitive coverage to avoid liability

        Please note: Some queries may be in form of a situation, use the context provided to answer the questions which follows the sitation and justify  """
    ),
    role=MessageRole.SYSTEM,
)

TEXT_QA_PROMPT_TMPL_MSGS = [
    TEXT_QA_SYSTEM_PROMPT,
    ChatMessage(
        content=(
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, "
            "answer the query.\n"
            "Query: {query_str}\n"
            "Answer: "
        ),
        role=MessageRole.USER,
    ),
]

CHAT_TEXT_QA_PROMPT = ChatPromptTemplate(message_templates=TEXT_QA_PROMPT_TMPL_MSGS)