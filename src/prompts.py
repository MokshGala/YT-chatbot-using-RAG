"""
src/prompts.py
--------------
Defines all LangChain PromptTemplates used by the RAG pipeline.

Keeping prompts in a dedicated module makes them easy to iterate on,
A/B test, or swap without touching pipeline logic.
"""

from langchain_core.prompts import PromptTemplate


def get_rag_prompt() -> PromptTemplate:
    """
    Return the primary RAG prompt template.

    The prompt instructs the model to:
    - Answer strictly from the provided video transcript context
    - Acknowledge when the context is insufficient rather than hallucinate
    - Be concise and direct

    Returns:
        A LangChain PromptTemplate with input variables: context, question.
    """
    template = """You are an AI assistant helping users understand YouTube video content.

You have been given the following transcript excerpt(s) from the video as context:

--- TRANSCRIPT CONTEXT ---
{context}
--- END CONTEXT ---

User Question: {question}

Instructions:
- Answer the question based ONLY on the transcript context provided above.
- If the context does not contain enough information to answer the question, \
say "I couldn't find information about that in this video's transcript."
- Be concise and direct in your answer.
- If quoting from the transcript, keep quotes brief and relevant.
- Do not fabricate information or draw on knowledge outside the transcript.

Answer:"""

    return PromptTemplate(
        template=template,
        input_variables=["context", "question"],
    )
