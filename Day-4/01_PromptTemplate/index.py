from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')


paper_input = "Attention is All You Need"
style_input = "Beginner-Friendly"
length_input = "Short (1-2 paragraphs)"


prompt_template = """
    Please summarize the research paper titled "{paper_input}" with the following specifications:
        Explanation Style: {style_input}  
        Explanation Length: {length_input}  
        1. Mathematical Details:  
            - Include relevant mathematical equations if present in the paper.  
            - Explain the mathematical concepts using simple, intuitive code snippets where applicable.  
        2. Analogies:  
            - Use relatable analogies to simplify complex ideas.  
    If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.  
    Ensure the summary is clear, accurate, and aligned with the provided style and length.
"""


template = PromptTemplate(
    template = prompt_template,
    input_variables = ["paper_input", "style_input", "length_input"],
    validate_template = True
)

prompt = template.invoke({
    "paper_input" : paper_input,
    "style_input" : style_input,
    "length_input" : length_input
})

result = model.invoke(prompt)

print(result.content)