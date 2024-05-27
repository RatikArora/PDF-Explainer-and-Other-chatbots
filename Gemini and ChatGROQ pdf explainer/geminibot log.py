    # how to run this ?
    # answer : streamlit run second.py



import streamlit as st
from dotenv import load_dotenv
load_dotenv()

import os
import pandas as pd

import google.generativeai as genai
import requests

genai.configure(api_key=os.getenv('Google_Api_Key'))
# Function to interact with Gemini API
def get_gemini_res(question,prompt,log):
    model = genai.GenerativeModel('gemini-pro')
    res = model.generate_content([prompt[0], question, log])
    print("response received was : ",res)
    if hasattr(res, 'text'):
        return res.text
    else:
        error_message = "Error: Unable to retrieve text from the response."
        return error_message
    
prompt = [""" 
        As a sophisticated chatbot designed to engage in meaningful conversations, I rely on your input to guide our interaction. Whether you have questions
        , thoughts,
        or just want to chat, feel free to express yourself openly. 
        Remember, our previous conversations are valuable insights into our ongoing dialogue, so don't hesitate to refer back to them for context. 
        Let's embark on another enriching exchange together!
        
"""]

def save_session_messages(messages):
    with open("session.txt", "w") as file:
        for message in messages:
            file.write(f"{message['role']}: {message['content']}\n")

def load_session_messages():
    messages = []
    if os.path.exists("session.txt"):
        with open("session.txt", "r") as file:
            role = None
            content = ""
            for line in file:
                line = line.strip()
                if line.startswith("user:"):
                    # Store previous message
                    if role is not None and content:
                        messages.append({"role": role, "content": content.strip()})
                    # Set role to user and reset content
                    role = "user"
                    content = line[len("user:"):].strip()
                elif line.startswith("assistant:"):
                    # Store previous message
                    if role is not None and content:
                        messages.append({"role": role, "content": content.strip()})
                    # Set role to assistant and reset content
                    role = "assistant"
                    content = line[len("assistant:"):].strip()
                else:
                    # Append to content
                    content += "\n" + line.strip()
            # Append the last message
            if role is not None and content:
                messages.append({"role": role, "content": content.strip()})
    return messages

def main():
    st.title("Gemini Chatbot")

    # Initialize list to store messages
    if "messages" not in st.session_state:
        st.session_state.messages = load_session_messages()

    # User input
    user_input = st.chat_input("Chat Gemini: ")

    # Check for new user input
    if user_input and (not st.session_state.messages or st.session_state.messages[-1]["content"] != user_input):
        # Add user input to messages
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Get Gemini response
        log = str(st.session_state.messages)
        response = get_gemini_res(user_input, prompt, log)

        # Add Gemini response to messages
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Save session messages
        save_session_messages(st.session_state.messages)

    # Display previous messages
    for message in st.session_state.messages:
        with st.container():
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


if __name__ == "__main__":
    main()



