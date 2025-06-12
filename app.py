import streamlit as st
from model import load_models
from prompts import create_prompts
from chains import create_chains
from state import HiringState
from conversation import handle_conversation
from utils import analyze_sentiment, generate_pdf

# Custom CSS for a visually appealing UI
st.markdown("""
    <style>
    .main {
        background-color: #f0f4f8;
        color: #333;
    }
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
    .stChatMessage.user {
        background-color: #e6f3ff;
        border: 1px solid #cce5ff;
    }
    .stChatMessage.assistant {
        background-color: #d9f2e6;
        border: 1px solid #b3e6cc;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        border-right: 1px solid #ddd;
    }
    h1, h3 {
        color: #2c3e50;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .stTextInput>div>input {
        border-radius: 5px;
        border: 1px solid #ccc;
    }
    </style>
""", unsafe_allow_html=True)

# Title and introduction
st.title("TalentScout Hiring Assistant")
st.write("Welcome to TalentScout's initial screening process. Interact with our chatbot below to begin!")

# Sidebar with information
st.sidebar.title("About TalentBot")
st.sidebar.write("This chatbot assists with the initial screening of candidates for technology placements at TalentScout.")
st.sidebar.write("It will collect your information and ask technical questions based on your tech stack.")
st.sidebar.write("**Tip**: Type 'exit' to end the conversation or 'query: your question' during technical questions for clarification.")

# Load models with caching
@st.cache_resource
def get_models():
    return load_models()

info_llm, question_llm, sentiment_pipeline = get_models()

# Create prompts and chains
info_gathering_prompt, tech_question_prompt, closing_prompt, relevance_prompt, revision_prompt = create_prompts()
info_gathering_chain, tech_question_chain, closing_chain, relevance_chain, revision_chain = create_chains(
    info_llm, question_llm, info_gathering_prompt, tech_question_prompt, closing_prompt, relevance_prompt, revision_prompt
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'state' not in st.session_state:
    st.session_state.state = HiringState()
    response, st.session_state.state = handle_conversation(
        "", st.session_state.state, tech_question_chain, relevance_chain, revision_chain, sentiment_pipeline
    )
    st.session_state.messages.append({"role": "assistant", "content": response})

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    response, st.session_state.state = handle_conversation(
        user_input, st.session_state.state, tech_question_chain, relevance_chain, revision_chain, sentiment_pipeline
    )
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# Show summary and PDF download when conversation ends and all 5 questions are answered
if st.session_state.state.stage == "closing" and len(st.session_state.state.answers) >= 5:
    st.write("### Conversation Summary")
    st.write(f"**Total Interactions**: {len(st.session_state.state.conversation_log)}")
    
    sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    sentiment_scores = []
    for log in st.session_state.state.conversation_log:
        if log.startswith("Sentiment:"):
            parts = log.split(':')
            if len(parts) > 1:
                sentiment = parts[1].strip().split()[0]
                if sentiment in sentiment_counts:
                    sentiment_counts[sentiment] += 1
                if '(' in parts[1]:
                    score_str = parts[1].split('(')[1].split(')')[0]
                    try:
                        sentiment_scores.append(float(score_str))
                    except:
                        pass
    if sentiment_scores:
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        st.write("#### Sentiment Analysis")
        st.write(f"- **Positive Responses**: {sentiment_counts['POSITIVE']}")
        st.write(f"- **Negative Responses**: {sentiment_counts['NEGATIVE']}")
        st.write(f"- **Neutral Responses**: {sentiment_counts['NEUTRAL']}")
        st.write(f"- **Average Sentiment Score**: {avg_sentiment:.2f}")
        st.write(f"- **Overall Tone**: {'Positive' if avg_sentiment > 0.6 else 'Negative' if avg_sentiment < 0.4 else 'Neutral'}")

    # Provide PDF download option
    st.write("### Download Your Assessment")
    pdf_buffer = generate_pdf(st.session_state.state.tech_questions, st.session_state.state.answers)
    st.download_button(
        label="Download Assessment PDF",
        data=pdf_buffer,
        file_name="assessment.pdf",
        mime="application/pdf"
    )

    if st.button("Start New Conversation"):
        st.session_state.messages = []
        st.session_state.state = HiringState()
        response, st.session_state.state = handle_conversation(
            "", st.session_state.state, tech_question_chain, relevance_chain, revision_chain, sentiment_pipeline
        )
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()