from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')


prompt1 = PromptTemplate(
    template = "Generate a tweet about following : \n {topic}",
    input_variables = ["topic"]
)

prompt2 = PromptTemplate(
    template = "Generate a Linkedin post about following : \n {topic}",
    input_variables = ["topic"]
)

parser = StrOutputParser()

chain = RunnableParallel({
    'tweet' : RunnableSequence(prompt1, model, parser),
    'linkedin' : RunnableSequence(prompt2, model, parser)
})

result = chain.invoke({"topic" : "AI"})

print(result["tweet"])