from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableBranch, RunnableLambda
from dotenv import load_dotenv


load_dotenv()


model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')


prompt1 = PromptTemplate(
    template = "Write a detailed report on following : \n {topic}",
    input_variables = ["topic"]
)


prompt2 = PromptTemplate(
    template = "Summarize the following text : \n {text}",
    input_variables = ["text"]
)


parser = StrOutputParser()


report_gen_chain = prompt1 | model | parser


branch_chain = RunnableBranch(
    (lambda x : len(x.split()) > 300, prompt2 | model | parser),
    RunnablePassthrough()
)


chain = RunnableSequence(report_gen_chain, branch_chain)


result = chain.invoke({"topic" : "Bitcoin"})


print(result)