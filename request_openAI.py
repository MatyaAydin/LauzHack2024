from OPEN_AI_KEY import key
from langchain.llms import OpenAI




llm = OpenAI(temperature=0, openai_api_key=key)

prompt_request = "How many boats passed from 5pm to 8pm?"

prompt = f"""
Given the following request:
{prompt_request}


Answer only in the following format:
count-start-end

where start and end are the starting and ending hours the user want the count to end. Use a 24 hours scale
"""

answer = llm.invoke(prompt)


answer_to_question = 5

prompt_answer = f"""
Someones asked the following: {prompt_request}

We got the following answer: {answer_to_question}

Format a nice answer to the question
"""

final_answer = llm.invoke(prompt_answer)



print(final_answer)