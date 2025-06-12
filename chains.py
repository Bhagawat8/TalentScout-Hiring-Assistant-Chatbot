from langchain.chains import LLMChain

def create_chains(info_llm, question_llm, info_gathering_prompt, tech_question_prompt, closing_prompt, relevance_prompt, revision_prompt):
    """Create and return all LLM chains for the chatbot."""
    info_gathering_chain = LLMChain(llm=info_llm, prompt=info_gathering_prompt, verbose=False)
    tech_question_chain = LLMChain(llm=question_llm, prompt=tech_question_prompt, verbose=False)
    closing_chain = LLMChain(llm=info_llm, prompt=closing_prompt, verbose=False)
    relevance_chain = LLMChain(llm=info_llm, prompt=relevance_prompt, verbose=False)
    revision_chain = LLMChain(llm=question_llm, prompt=revision_prompt, verbose=False)
    return info_gathering_chain, tech_question_chain, closing_chain, relevance_chain, revision_chain