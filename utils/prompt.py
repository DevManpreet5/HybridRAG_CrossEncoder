def format_prompt(query, contexts):
    context_block = "\n---\n".join(contexts)
    return f"Answer the following query using the context below:\n\nContext:\n{context_block}\n\nQuery: {query}\nAnswer:"