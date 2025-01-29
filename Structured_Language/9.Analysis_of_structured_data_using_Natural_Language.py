import pandas as pd 
from dotenv import load_dotenv
from langchain.agents.agent_types import AgentType 
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import OpenAI 
import streamlit as st
import warnings
warnings.filterwarnings("ignore")


def main():
    load_dotenv()
    df = pd.read_csv("attrition.csv")

    st.set_page_config(
        page_title = "Documentation Chatbot",
        page_icon = ":books",
    )

    st.title("Data Analysis Chatbot")
    st.subheader("Unvover Insights from Data")
    st.markdown(
        """
        This chatbot was created to answer questions from a dataset from the organization. Ask a question and the chatbot will
        respond with appropriate Analysis
        """
    )

    st.write(df.head())

    user_question = st.text_input("Ask your question about the data")


    agent = create_csv_agent(
        OpenAI(temperature=0),
        "attrition.csv",
        verbose = True,
        agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        allow_dangerous_code=True,
        handle_parsing_errors=True,
    )

    # print(agent.agent.llm_chain.prompt.template)

    answer = agent.invoke(user_question)
    st.write(answer)

if __name__ == "__main__":
    main()





