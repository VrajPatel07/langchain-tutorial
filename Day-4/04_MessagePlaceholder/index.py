from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


chat_template = ChatPromptTemplate([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("history"),
    ("human", "{question}")
])


prompt = chat_template.invoke({
    "history" : [("human", "what's 5 + 2"), ("ai", "5 + 2 is 7")],
    "question" : "now multiply that by 4"
})

print(prompt)