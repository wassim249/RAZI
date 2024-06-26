import streamlit as st


# edit streamlit page config
st.set_page_config(
    page_title="RAZI.AI",
    page_icon="👨🏼‍⚕️",
    layout="centered",
    initial_sidebar_state="auto",
)

st.title("👨🏼‍⚕️ RAZI.AI")
st.markdown("This is a demo of a chatbot that helps you detect a mental illness. It is powered by OpenAI's GPT-3 API and a fine-tuned version of Google's Bert model.")
st.markdown("Made by [Wassim EL BAKKOURI](#wassim.elbakkouri@yahoo.com) & [Fatima Zahra MOUMENE](#moumene.fatimazahra2000@gmail.com)")
st.markdown('<style>' + open('./styles/style.css').read() + '</style>', unsafe_allow_html=True)


from backend.agent import RAZIAgent
from langchain_core.messages import AIMessage

agent = RAZIAgent().get_agent()

def app():
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    for message in st.session_state.messages:
        print(st.chat_message("RAZI"))
        with st.chat_message("RAZI" if message["role"] == "RAZI" else "user"):
            st.markdown(message["role"])
            st.markdown(message["content"])
            
            

    
    if prompt := st.chat_input("Feel free to describe your situation"):
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("RAZI"):
            msg_placeholder = st.empty()
            
            _,history,output = agent.invoke({'input': prompt}).values()
            print(history)
            for message in history:
                print(message)
                st.session_state.messages.append({
                    'role' : 'RAZI' if isinstance(message,AIMessage) else 'user',
                    'content': message.content,
                    
                })
            msg_placeholder.markdown(output)
            


if __name__ == "__main__":
    app()