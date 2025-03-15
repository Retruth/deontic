from enum import Enum
from dataclasses import dataclass

# Classes
# SystemPrompt: prompt for the system
# QuestionType: type of question
# AnswerType: type of answer
# PromptGenerator: generate prompts

@dataclass
class SystemPrompt:
    prompt_text: str
    
class QuestionType(Enum):
    GENERAL = "general"
    STRICT = "strict"
    EXPLICIT = "explicit"

class AnswerType(Enum):
    BINARY = "binary"
    MULTIPLE = "multiple"
    REGRESSION = "regression"
    
class PromptGenerator:
    def __init__(self):
        pass
    
    def generate_prompts(self, question_type: QuestionType, answer_type: AnswerType):
        pass
    
    def get_ctx_inp(self, question_type: QuestionType, context: str, sentence: str):
        # Initialize a prompt
        common_prompt = "Answer with 1 if it's a deontic sentence, 0 if not.you must answer with 1 or 0.\n" \
                        f"Sentence: Context: {context}\nInput: {sentence}\n\nAnswer: "
        question_prompts = {
            QuestionType.GENERAL: SystemPrompt(
                prompt_text="Determine if the following sentence is deontic by considering the context and the semantic meaning."
                ),
            QuestionType.STRICT: SystemPrompt(
                prompt_text="Determine whether this sentence mandates compliance in all cases by considering the context and the semantic meaning.",
                ),
            QuestionType.EXPLICIT: SystemPrompt(
                prompt_text="Determine whether the following sentence is an obligation based on its context and semantic meaning.",
                ),
        }
        final_prompt= common_prompt + question_prompts[question_type].prompt_text
        return final_prompt
            