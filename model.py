from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
import torch

def load_models():
    """Load transformer models and set up pipelines for info gathering, question generation, and sentiment analysis."""
    model_name = "Qwen/Qwen3-4B"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        pad_token='<|endoftext|>',
        padding_side='left'
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    info_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.3,
        top_p=0.9,
        repetition_penalty=1.3,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id
    )
    
    question_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        temperature=1,
        top_p=0.95,
        repetition_penalty=1.3,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id
    )
    
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True
    )
    
    info_llm = HuggingFacePipeline(pipeline=info_pipeline)
    question_llm = HuggingFacePipeline(pipeline=question_pipeline)
    
    return info_llm, question_llm, sentiment_pipeline