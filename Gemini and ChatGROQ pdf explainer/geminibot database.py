import sqlite3
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
# Function to create the SQLite database and table if they don't exist
def create_database_table():
    conn = sqlite3.connect('messages.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS messages
                 (id INTEGER PRIMARY KEY, role TEXT, content TEXT)''')
    conn.commit()
    conn.close()

create_database_table()

# Function to insert a message into the database
def insert_message(role, content):
    conn = sqlite3.connect('messages.db')
    c = conn.cursor()
    c.execute("INSERT INTO messages (role, content) VALUES (?, ?)", (role, content))
    conn.commit()
    conn.close()

# Function to load messages from the database
def load_session_messages():
    messages = []
    conn = sqlite3.connect('messages.db')
    c = conn.cursor()
    c.execute("SELECT * FROM messages")
    rows = c.fetchall()
    for row in rows:
        messages.append({"role": row[1], "content": row[2]})
    conn.close()
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

        # Save user input
        insert_message("user", user_input)
        # Save Gemini response
        insert_message("assistant", response) 
        # Display previous messages

    for message in st.session_state.messages:
        with st.container():
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


if __name__ == "__main__":
    main()
