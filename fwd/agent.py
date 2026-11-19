import os
import pandas as pd
import warnings
from dotenv import load_dotenv
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*langchain-experimental.*")

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
web_api_key = os.getenv("TAVILY_API_KEY")

model = ChatGroq(
    model_name="qwen/qwen3.8-27b",  
    api_key=api_key,
    temperature=0
)
web_search = TavilySearch(max_results=3)

@tool
def web_search_tool(query: str) -> str:
    """Use this tool to search the internet for Samsung's live share price, global market news, or latest tech updates."""
    return str(web_search.invoke(query))

try:
    df = pd.read_csv("samsung_inventory.csv")
except Exception:
    print("Error: 'samsung_inventory.csv' file nahi mili. Kripya file check karein.")
    exit()

csv_agent = create_pandas_dataframe_agent(
    model, 
    df, 
    verbose=False, 
    allow_dangerous_code=True
)

@tool
def samsung_data_tool(query: str) -> str:
    """Use this tool to get information from the local CSV about Samsung products, mobiles, appliances, their prices, and stock status."""
    response = csv_agent.invoke({"input": query})
    return str(response["output"])

my_tools = [samsung_data_tool, web_search_tool]

system_msg = """You are a smart AI assistant for an official Samsung Electronics Store.
Guidelines:
1. First Priority: Use 'samsung_data_tool' for local store inventory, product availability, warranties, and internal prices (like Galaxy S24, TVs, etc.).
2. Second Priority: Use 'web_search_tool' for real-time market data, Samsung's stock/share price today, or upcoming global launches.
3. Stay on topic: You are a Samsung tech expert. If a user asks completely unrelated questions, politely bring them back to Samsung products."""

agent = create_agent(
    model=model,
    tools=my_tools,
    system_prompt=system_msg
)

print("\n📱 Samsung Store AI Agent is Ready!")
print("Type 'exit' to quit.")
while True:
    que = input("\nUser: ")

    if que.lower() in {"exit", "quit"}:
        print("Agent: Thank you for visiting the Samsung Store! Goodbye.")
        break

    if not que.strip():
        continue

    try:
        response = agent.invoke({
            "messages": [
                {
                    "role": "user", "content": que
                }
            ]
        })
        
        ans = response["messages"][-1].content
        print("\nAgent:", ans)
        
    except Exception as error:
        print("An error occurred:", error)