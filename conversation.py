from utils import clean_response, extract_questions, format_tech_stack, analyze_sentiment
from prompts import create_prompts
from chains import create_chains

def generate_tech_questions(state, tech_question_chain):
    """Generate technical questions based on candidate data."""
    required_fields = ['tech_stack', 'years_experience', 'desired_position']
    if any(state.candidate_data[field] is None for field in required_fields):
        state.stage = "closing"
        return "We don't have enough information to generate questions. Thank you for your time.", state

    response = tech_question_chain.invoke({
        "tech_stack": ", ".join(state.candidate_data["tech_stack"]),
        "years_experience": state.candidate_data["years_experience"],
        "desired_position": state.candidate_data["desired_position"]
    })['text']

    cleaned_response = clean_response(response)
    state.tech_questions = extract_questions(cleaned_response)

    if not state.tech_questions:
        position = state.candidate_data["desired_position"] or "this position"
        tech = state.candidate_data["tech_stack"][0] if state.candidate_data["tech_stack"] else "your primary technology"
        state.tech_questions = [
            f"1. What experience do you have with {tech}?",
            f"2. Describe a challenging project you've worked on using {tech}.",
            f"3. How would you debug a performance issue in a {tech} application?",
            f"4. What best practices do you follow when working with {tech}?",
            f"5. How does your experience align with the requirements for {position}?"
        ]

    state.log_interaction(f"Generated {len(state.tech_questions)} technical questions")
    return None, state

def handle_conversation(user_input, state, tech_question_chain, relevance_chain, revision_chain, sentiment_pipeline):
    """Handle the conversation flow with the candidate."""
    if user_input.lower() in ['exit', 'quit', 'stop', 'end']:
        closing_response = "Thank you for your time. We'll be in touch soon!"
        state.memory.save_context({"input": user_input}, {"output": closing_response})
        state.log_interaction(f"Assistant: {closing_response}")
        return closing_response, state

    if user_input:
        state.log_interaction(f"User: {user_input}")
        if state.stage not in ["greeting", "awaiting_start", "info_gathering"]:
            sentiment = analyze_sentiment(user_input, sentiment_pipeline)
            state.log_interaction(f"Sentiment: {sentiment['label']} ({sentiment['score']:.2f})")

    state.memory.save_context({"input": user_input}, {"output": ""})

    if state.stage == "technical_interview" and user_input.lower().startswith("query:"):
        query_text = user_input[len("query:"):].strip()
        current_question = state.tech_questions[state.current_question_idx]
        
        relevance_response = relevance_chain.invoke({
            "current_question": current_question,
            "query_text": query_text
        })['text'].lower().strip()
        
        if "yes" in relevance_response:
            revised_question = revision_chain.invoke({
                "current_question": current_question,
                "query_text": query_text
            })['text'].strip()
            revised_question = clean_response(revised_question)
            state.tech_questions[state.current_question_idx] = revised_question
            state.log_interaction(f"Revised Question: {revised_question}")
            response = f"Thank you for your query. Here's the revised question:\n\n{revised_question}"
        else:
            response = "Your query doesn't appear relevant to the current question. Please answer the original question."
        
        state.memory.save_context({"input": user_input}, {"output": response})
        state.log_interaction(f"Assistant: {response}")
        return response, state

    if state.stage == "greeting":
        greeting = "Hello! I'm TalentBot from TalentScout. I'll guide you through our initial screening process. "\
                   "If you're ready to begin, please type 'Start'.\n\n"\
                   "You can type 'exit' at any time to end the conversation.\n"\
                   "During technical questions, you can request clarification by typing 'query: your question'."
        state.stage = "awaiting_start"
        state.log_interaction("System: Initial greeting")
        return greeting, state

    elif state.stage == "awaiting_start":
        if user_input.lower() == "start":
            state.stage = "info_gathering"
            state.log_interaction("User started the process")
            first_field = state.get_current_field()
            return state.FIELD_PROMPTS[first_field], state
        else:
            return "Please type 'Start' when you're ready to begin the screening process.", state

    elif state.stage == "info_gathering":
        current_field = state.get_current_field()
        if current_field in state.FIELD_VALIDATORS:
            validator = state.FIELD_VALIDATORS[current_field]
            if not validator(user_input):
                return state.FIELD_ERRORS[current_field], state
        state.record_response(current_field, user_input)
        if state.next_field():
            next_field = state.get_current_field()
            return state.FIELD_PROMPTS[next_field], state
        else:
            state.stage = "tech_stack_collection"
            state.log_interaction("System: Collecting tech stack")
            return state.FIELD_PROMPTS["tech_stack"], state

    elif state.stage == "tech_stack_collection":
        tech_stack = format_tech_stack(user_input)
        state.candidate_data["tech_stack"] = tech_stack
        state.log_interaction(f"Tech stack: {', '.join(tech_stack)}")
        state.stage = "technical_interview"
        msg, state = generate_tech_questions(state, tech_question_chain)
        if msg:
            return msg, state
        if state.tech_questions:
            first_question = state.tech_questions[0].split('.', 1)[1].strip()
            response = f"Thank you! Let's begin the technical assessment. You'll be asked {len(state.tech_questions)} questions.\n\n"\
                       f"Question 1: {first_question}\n\n"\
                       "If you need clarification on any question, type 'query: your question'."
        else:
            state.stage = "closing"
            response = "We've completed the initial screening. Thank you for your time!"
        state.memory.save_context({"input": user_input}, {"output": response})
        state.log_interaction(f"Assistant: {response}")
        return response, state

    elif state.stage == "technical_interview":
        if state.current_question_idx < len(state.tech_questions):
            question = state.tech_questions[state.current_question_idx]
            state.log_interaction(f"Question {state.current_question_idx+1}: {question}")
            if not user_input.lower().startswith("query:"):
                state.log_interaction(f"Answer: {user_input[:200]}...")
                state.answers.append(user_input)  # Store the answer
                state.current_question_idx += 1
            if state.current_question_idx < len(state.tech_questions):
                next_q = state.tech_questions[state.current_question_idx].split('.', 1)[1].strip()
                response = f"Question {state.current_question_idx + 1}: {next_q}"
            else:
                state.stage = "closing"
                state.log_interaction("Completed technical assessment")
                response = "Thank you for completing the assessment! Our team will review your answers and contact you soon."
            state.memory.save_context({"input": user_input}, {"output": response})
            state.log_interaction(f"Assistant: {response}")
            return response, state

    if state.stage == "closing":
        response = "Thank you again! Our team will review your application shortly."
        state.memory.save_context({"input": user_input}, {"output": response})
        state.log_interaction(f"Assistant: {response}")
        return response, state

    response = "I'm here to assist with your job application. Could you please rephrase that?"
    state.memory.save_context({"input": user_input}, {"output": response})
    state.log_interaction(f"Assistant: {response}")
    return response, state