import openai
import streamlit as st
from scripts.utils import build_prompt , load_model , predict


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



# load the OpenAI API key
openai.api_key = st.secrets["openai_secret"]

# load the model and tokenizer from the secrets
model , tokenizer , label_encoder = load_model(
    config_file = st.secrets["config_file"],
    model_file = st.secrets["model_file"],
    label_encoder_file = st.secrets["label_encoder_file"],
    tokenizer_folder = st.secrets["tokenizer_folder"]
    )

def app():
    # check if the user has selected a model
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    # display the model selection widget
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # display the messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # check if the user has entered a prompt
    if prompt := st.chat_input("Feel free to describe your situation"):
        # append the prompt to the messages
        st.session_state.messages.append({"role": "user", "content": prompt})
        # send the prompt to the API
        with st.chat_message("user"):
            st.markdown(prompt)
        # send the response to the API
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            prediction , _ = predict(model,tokenizer,label_encoder,prompt)
            prompt = build_prompt(prompt, prediction)

            full_response = f"you are diagnosed with {prediction}\n\n"

            # send the prompt to the API
            for response in openai.ChatCompletion.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": prompt}
                    for m in st.session_state.messages
                ],
                stream=True,
            ):

                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        # append the full response to the messages
        st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    app()