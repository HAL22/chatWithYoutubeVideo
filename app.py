import streamlit as st
import qa
import time

st.title("Query any Youtube video")

submitted = False
with st.form(key='txt_input'):
    txt_input = st.text_area('Enter youtube url', '', height=80)
    submit_button = st.form_submit_button(label='Enter')
    if submit_button:
        if 'url' not in st.session_state:
            st.session_state['url'] = txt_input
        qa.load_pinecone(txt_input,400)
        submitted = True

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            qa.get_video(st.session_state['url'],message["start"])

# Accept user input
user_ans = ""
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    user_ans = prompt
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display assistant response in chat message container
with st.chat_message("assistant"):
    message_placeholder = st.empty()
    full_response = ""
    start = 0
    if user_ans!=None and len(user_ans)>0:
        st.write(user_ans)
        assistant_response,start = qa.qa_answer(user_ans)

    else:
        assistant_response = "No answer"


    if user_ans!=None and len(user_ans)>0:
        # Simulate stream of response with milliseconds delay
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        qa.get_video(st.session_state['url'],start)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response, "start":start})


          