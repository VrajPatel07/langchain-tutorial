from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableLambda, RunnablePassthrough, RunnableParallel
from dotenv import load_dotenv


load_dotenv()


model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')


def word_count(text):
    return len(text.split())


prompt = PromptTemplate(
    template = "Write a joke about following : \n {topic}",
    input_variables = ["topic"]
)


parser = StrOutputParser()


joke_gen_chain = RunnableSequence(prompt, model, parser)


parallel_chain = RunnableParallel({
    "joke" : RunnablePassthrough(),
    "word_count" : RunnableLambda(word_count)
})


chain = RunnableSequence(joke_gen_chain, parallel_chain)


result = chain.invoke({'topic':'AI'})


print(result["word_count"])
# print(result["joke"])