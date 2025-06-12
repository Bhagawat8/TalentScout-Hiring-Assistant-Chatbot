from langchain.memory import ConversationBufferMemory
import re
import json

class HiringState:
    """Manages the state of the hiring conversation."""
    FIELDS = [
        "full_name",
        "email",
        "phone",
        "years_experience",
        "desired_position",
        "current_location"
    ]
    FIELD_DISPLAY = {
        "full_name": "full name",
        "email": "email address",
        "phone": "phone number",
        "years_experience": "years of professional experience",
        "desired_position": "desired position",
        "current_location": "current location",
        "tech_stack": "tech stack (comma-separated list of technologies)"
    }
    FIELD_PROMPTS = {
        "full_name": "What is your full name?",
        "email": "What is your email address?",
        "phone": "What is your phone number?",
        "years_experience": "How many years of professional experience do you have?",
        "desired_position": "What is your desired position?",
        "current_location": "What is your current location?",
        "tech_stack": "Please list your technical skills (comma-separated):"
    }
    FIELD_VALIDATORS = {
        "email": lambda x: re.match(r"[^@]+@[^@]+\.[^@]+", x) is not None,
        "phone": lambda x: re.match(r"^\+?[0-9\s\-\(\)]{7,}$", x) is not None,
        "years_experience": lambda x: re.match(r"^\d+$", x) is not None
    }
    FIELD_ERRORS = {
        "email": "Please enter a valid email address (e.g., name@example.com).",
        "phone": "Please enter a valid phone number (e.g., +1 123-456-7890).",
        "years_experience": "Please enter a valid number of years (e.g., 5)."
    }

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the conversation state to initial values."""
        self.stage = "greeting"
        self.current_field_idx = 0
        self.candidate_data = {field: None for field in self.FIELDS}
        self.candidate_data["tech_stack"] = None
        self.tech_questions = []
        self.current_question_idx = 0
        self.memory = ConversationBufferMemory()
        self.conversation_log = []
        self.answers = []  # Added to store user answers to technical questions

    def get_current_field(self):
        """Get the current field being collected."""
        if self.current_field_idx < len(self.FIELDS):
            return self.FIELDS[self.current_field_idx]
        return None

    def record_response(self, field, value):
        """Record the candidate's response for a given field."""
        if field in self.FIELDS:
            self.candidate_data[field] = value
            self.log_interaction(f"User provided {field}: {value}")

    def next_field(self):
        """Move to the next field in the info gathering process."""
        if self.current_field_idx < len(self.FIELDS) - 1:
            self.current_field_idx += 1
            return True
        return False

    def log_interaction(self, message):
        """Log an interaction in the conversation."""
        self.conversation_log.append(message)

    def to_dict(self):
        """Convert the state to a dictionary for debugging or storage."""
        return {
            "stage": self.stage,
            "current_field": self.get_current_field(),
            "candidate_data": self.candidate_data,
            "tech_questions": self.tech_questions,
            "current_question_idx": self.current_question_idx,
            "answers": self.answers  
        }