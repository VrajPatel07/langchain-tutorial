from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

prompt1 = PromptTemplate(
    template = "Write a joke about following : \n {topic}",
    input_variables = ["topic"]
)

prompt2 = PromptTemplate(
    template = "Explain the following joke : \n {joke}",
    input_variables = ["joke"]
)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    "joke" : RunnablePassthrough(),
    "explanation" : RunnableSequence(prompt2, model, parser)
})

chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = chain.invoke({"topic" : "cricket"})

print(result["joke"])
# print(result["explanation"])