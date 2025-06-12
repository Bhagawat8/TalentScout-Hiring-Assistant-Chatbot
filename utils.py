import re
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def clean_response(response):
    """Clean and format the model response."""
    response = re.sub(r'<\|im_start\|>.*?<\|im_end\|>', '', response, flags=re.DOTALL)
    response = re.sub(r'<\|.*?\|>', '', response, flags=re.DOTALL)
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    if "assistant" in response.lower():
        parts = re.split(r'assistant', response, flags=re.IGNORECASE)
        if len(parts) > 1:
            response = parts[-1].strip()
    response = re.sub(r'^\d+[\.\)\s]*', '', response).strip()
    response = re.sub(r'\n+', '\n', response).strip()
    return response

def extract_questions(response):
    """Extract generated questions from model response."""
    questions = []
    lines = response.split('\n')
    for line in lines:
        match = re.match(r'(\d+)\.?\s*(.*)', line.strip())
        if match:
            question_num = int(match.group(1))
            question_text = match.group(2).strip()
            if len(question_text) > 20 and '?' in question_text:
                questions.append(f"{question_num}. {question_text}")
    if len(questions) < 5:
        pattern = r'\d+\.\s*([^\n?]+\??)'
        alt_questions = re.findall(pattern, response)
        questions = [f"{i+1}. {q.strip()}" for i, q in enumerate(alt_questions[:5])]
    return questions[:5]

def format_tech_stack(tech_input):
    """Format the tech stack input into a list."""
    return [tech.strip() for tech in tech_input.split(',') if tech.strip()]

def analyze_sentiment(text, sentiment_pipeline):
    """Analyze the sentiment of the given text."""
    if len(text) < 3:
        return {"label": "NEUTRAL", "score": 0.0}
    try:
        result = sentiment_pipeline(text, truncation=True)[0]
        return {"label": result['label'], "score": float(result['score'])}
    except:
        return {"label": "ERROR", "score": 0.0}

def generate_pdf(questions, answers):
    """Generate a PDF of the technical assessment questions and answers."""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica", 12)
    y = 750
    c.drawString(100, y, "Technical Assessment")
    y -= 30
    for i, question in enumerate(questions):
        q_text = re.sub(r'^\d+\.\s*', '', question).strip()
        c.drawString(100, y, f"Question {i + 1}: {q_text}")
        y -= 20
        if i < len(answers):
            c.drawString(100, y, f"Answer: {answers[i]}")
        else:
            c.drawString(100, y, "Answer: Not provided")
        y -= 30
        if y < 50:
            c.showPage()
            y = 750
    c.save()
    buffer.seek(0)
    return buffer