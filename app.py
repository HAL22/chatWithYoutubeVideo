import streamlit as st
import qa
import time

st.title("Query any Youtube video")

submitted = False
with st.form(key='txt_input'):
    txt_input = st.text_area('Enter youtube url', '', height=80)
    submit_button = st.form_submit_button(label='Enter')
    if submit_button:
        qa.load_pinecone(txt_input,400)
        submitted = True

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display assistant response in chat message container
with st.chat_message("assistant"):
    message_placeholder = st.empty()
    full_response = ""
    if submitted and prompt!=" ":
        print(f"Here's the query: {prompt}")
        assistant_response = qa.qa_answer(prompt)
    else:
        assistant_response = ""

    # Simulate stream of response with milliseconds delay
    for chunk in assistant_response.split():
        full_response += chunk + " "
        time.sleep(0.05)
        # Add a blinking cursor to simulate typing
        message_placeholder.markdown(full_response + "▌")
    message_placeholder.markdown(full_response)
# Add assistant response to chat history
st.session_state.messages.append({"role": "assistant", "content": full_response})         

          