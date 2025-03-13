from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model_first = ChatOpenAI()

parser = StrOutputParser()

class Feedback(BaseModel):

    sentiment: Literal['positive', 'negative'] = Field(description = 'Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)


prompt1 = PromptTemplate(
    template = 'Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model_first | parser2

# results = classifier_chain.invoke({'feedback': 'This is an amazing smart phone'}).sentiment

# print(results)

prompt2 = PromptTemplate(
    template = 'Write an appropriate response to this positive feedback \n {feedback}',
    input_variables= ['feedback']
)

prompt3 = PromptTemplate(
    template = 'Write an appropriate response to this negative feedback \n {feedback}. Be sure to politely reply that we are kind of sorry and in an apologetic tone.',
    input_variables= ['feedback']
)






# branch_chain = RunnableBranch(
#     (condition1, chain1),
#     (condition2, chain2),
#     default chain
# )

# kind of if else statement

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model_first | parser),
    (lambda x:x.sentiment == 'negative', prompt3 | model_first | parser),
    RunnableLambda(lambda x : "could not find sentiment")
)



chain = classifier_chain | branch_chain

result1 = chain.invoke({'feedback': 'This is an amazing smart phone'})
result2 = chain.invoke({'feedback': 'This is a terrible product'})

print('------------- Process Flow ------------')
chain.get_graph().print_ascii()

print('\n\n\n------------- Output - Review Reply 1 ------------')
print(result1)

print('\n\n\n------------- Output - Review Reply 2 ------------')
print(result2)