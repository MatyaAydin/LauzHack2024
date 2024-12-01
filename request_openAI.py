from OPEN_AI_KEY import key
from langchain.llms import OpenAI
from os import listdir
from os.path import isfile, join
path = './csvs'
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]





llm = OpenAI(temperature=0, openai_api_key=key)

#Initial request caught from user

prompt_request = "How many boats passed from 5pm to 8pm on the 30th of november?"

prompt_about_time = f"""

Given the following prompt:

{prompt_request}

Extract only the information about time in a format such as from [starting hour] to [ending hour] on the [date]
"""

#Information about time only
answer_time = llm.invoke(prompt_about_time)

print(answer_time)


prompt = f"""
Given the following timestamp:
{answer_time}

Return all the csv names that are relevant to answer the question. Here are the csv. the format is day_month-starthour-endhour. Return the names only.
{onlyfiles}
"""

answer = llm.invoke(prompt)


print(answer)