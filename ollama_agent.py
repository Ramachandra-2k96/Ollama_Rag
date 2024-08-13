import json
from llm_axe import make_prompt, AgentType, OllamaChat, OnlineAgent, Agent
from datetime import datetime
import ollama

# This is the prompt used to guide the plan agent's decision-making
prompt = ''' 
        Determine whether or not the internet is required to answer the user's prompt. Carefully evaluate the question and consider the following:
        If the question can be answered using general knowledge, common sense, or logical reasoning without needing specific, up-to-date, or factual information, answer "no internet required".
        If the question requires specific, up-to-date, or factual information that is not stored in your training data, and needs to be searched online to provide an accurate answer, answer "internet required".
        Do not provide any additional responses, explanations, or answers. Do not elaborate or justify your decision. Simply respond with one of the two designated phrases: "no internet required" or "internet required".
'''


def stream_output(prompt):
    chat_history.append(make_prompt("user", user_input))
    response = ""
    stream = ollama.chat(model="llama3.1",messages=chat_history,stream=True)
    print("\033[95mAI: ",end='')
    for chunk in stream:
        content = chunk['message']['content']
        response+=content
        print(content,end='',flush=True)
    print("\033[0m\n")
    chat_history.append(make_prompt("assistant", response))
    
    
class FunctionCaller:
    def __init__(self):
        self.llm = OllamaChat(model="llama3.1") 
        self.agent = Agent(self.llm,agent_type=AgentType.GENERIC_RESPONDER)
        self.online_agent = OnlineAgent(self.llm)
        
    def internet(self, user_input, chat_history):
        resp = self.online_agent.search(user_input)
        return resp

    def no_internet(self, user_input, chat_history):
        stream_output(chat_history)
        

    def get_Datetime(self, user_input, chat_history):
        current_time = datetime.now().strftime("It's currently %A, %B %dth, %Y at %I:%M %p.")
        resp = self.agent.ask(user_input + " " + current_time, history=chat_history)
        return resp

    def analyse_Image(self, user_input, chat_history):
        image_paths = [f"{image}" for image in user_input.split() if "/" in image]
        normal_query = ' '.join([query for query in user_input.split() if "/" not in query])
        new_query = ' '.join([normal_query])
        new_query+=image_paths + "images: ["+"".join(image_paths)+"]"
        resp = self.agent.ask(new_query, history=chat_history)
        return resp   
    
    def get_function(self, plan):
        functions = {
            "no internet required": self.no_internet,
            "internet required": self.internet,
            "date and time": self.get_Datetime,
            "image analysis": self.analyse_Image
        }
        return functions.get(plan.strip(), None)

# Initialize the planning agent with a model and the custom prompt
plan_llm = OllamaChat(model="gemma2:2b-instruct-q2_K") 
plan_agent = Agent(plan_llm, custom_system_prompt=prompt)

chat_history = []
function_caller = FunctionCaller()

while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break    
    # Get the plan from the planning agent
    plan = plan_agent.ask("USERS INPUT: " + user_input.replace('?',"")).strip()
    # Determine the function based on the plan
    function = function_caller.get_function(plan.lower())
    if function:
        # Prepare a debug dictionary for better visualization
        debug_query = {
            "plan": plan,
            "user_input": user_input,
            "Function": function.__name__
        }
        pretty_json = json.dumps(debug_query, indent=4)
        print("\033[92mDebug: \n\n" + pretty_json + "\033[0m\n")

        # Call the appropriate function and get the result
        try:
            function(user_input=user_input, chat_history=chat_history)
        except Exception as e:
            print(f"\033[31m\nError: {str(e)}\033[0m")
    else:
        print(f"\033[31m\nError: No function found for the plan '{plan}'\033[0m")
