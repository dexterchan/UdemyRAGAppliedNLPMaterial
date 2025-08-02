# %%
from openai import OpenAI

client = OpenAI()
# %%
assistant = client.beta.assistants.create(
    name="Math Tutor",
    instructions="You are a personal math tutor. Write and run code to answer math questions.",
    tools=[{"type": "code_interpreter"}],
    model="gpt-4o",
)
# %%
thread = client.beta.threads.create()
# %%
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="I need to solve the equation `3x + 11 = 14`. Can you help me?",
)
# %%
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
    instructions="Please address the user as Jane Doe. The user has a premium account.",
)
# %% tourism
assistant = client.beta.assistants.create(
    name="Fukuoka Tour",
    instructions="""
                    You are a personal travel guide of Fukuoka, Japan.
                    You plan for a 1 week trip for a family of three:
                    - 2 adults
                    - 1 child of 6 years old
                    They are able to drive a car for the whole trip.
    
                   """,
    tools=[{"type": "code_interpreter"}],
    model="gpt-4o",
)
# %%
from openai import OpenAI

client = OpenAI()
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": "You are a personal travel guide of Fukuoka, Japan.",
        },
        {
            "role": "user",
            "content": """
                    You plan for a 1 week trip for a family of three:
                    - 2 adults
                    - 1 child of 6 years old
                    They are able to drive a car for the whole trip.
    
                   """,
        },
    ],
)
from pprint import pprint

pprint(completion.choices[0].message.content)

# %%

from openai import OpenAI

client = OpenAI()
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": "You are a personal travel guide of Birmingham, United Kingdom.",
        },
        {
            "role": "user",
            "content": """
                    You plan for a half day trip for a person from London.
    
                   """,
        },
    ],
)
from pprint import pprint

pprint(completion.choices[0].message.content)
# %%

from openai import OpenAI

client = OpenAI()
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": "You are a United Kingdom property tax consultant.",
        },
        {
            "role": "user",
            "content": """
                    UK property investment taxation. 
                    We use a company to hold the property for 200K. Then, two years later, we do re-mortgate of the property for 60% ~ 120K for stock investment. Do we need to pay any tax for that 120K?
    
                   """,
        },
    ],
)
from pprint import pprint

pprint(completion.choices[0].message.content)
# %%

from openai import OpenAI

client = OpenAI()
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": "You are a United Kingdom property tax consultant.",
        },
        {
            "role": "user",
            "content": """
Flat worth 400K under our company name (we personal lend 400K to the company to buy the flat)
rent received per month: $1000 

So this company has $1000 income and need to pay 19% tax to the government? After paying the 19% tax, the company will have $800 net income. If we draw out this $800 money to our personal account, is it consider as our personal income and we need to pay tax on it? Or this $800 is considered as a loan repayment for our lending of 400k so we don't need to pay any personal or other tax on this $800?
""",
        },
    ],
)
from pprint import pprint

pprint(completion.choices[0].message.content)
