
import os
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.hub import pull
import tensorflow as tf
import numpy as np
from transformers import BertTokenizer, TFBertForSequenceClassification , BertConfig
from pickle import load
import streamlit as st
from huggingface_hub import hf_hub_download


class RAZIAgent:
    def __init__(self):
        load_dotenv()
        self.prompt = pull("hwchase17/react")
        self.prompt.template = """You're name is RAZI,a mental health professional, you're role is to help the user detect his mental illness and provide him with the best advice to overcome it,
        you should use a friendly tone and always be compassionate,avoid using a robotic tone""" + \
        """Highlight the detected mental illness in bold""" + \
        """"Don't provide mental illness unless the user describes his situation,""" + \
        self.prompt.template
        
        
        self.llm = ChatOpenAI(model="gpt-4o",
                              openai_api_key=st.secrets["OPENAI_API_KEY"],
                              temperature=1)

        self.agent = create_react_agent(self.llm, [
            self.get_mentall_illness_category
        ], self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent,
                                            tools=[self.get_mentall_illness_category],
                                            verbose=True,
                                            handle_parsing_errors=True)
        
        self.agent_executor.memory = ConversationBufferMemory(k=2,
                                                              return_messages=True,
                                                              
                                                              )
    @staticmethod   
    def load_model(config_file  , label_encoder_file, tokenizer_folder):
        """
        Load the BERT model and tokenizer
        params:
            config_file: string
            model_file: string
            label_encoder_file: string
            tokenizer_folder: string
        return:
            model: TFBertForSequenceClassification
            tokenizer: BertTokenizer
            label_encoder: LabelEncoder
        """
        # Load the BERT model configuration
        config = BertConfig.from_json_file(config_file)

        # check if the model is already downloaded
        if not os.path.exists('../models/tf_model.h5'):

            with st.spinner("Please wait we are downloading the model..."):
                # download the model from the Hugging Face Hub and save it locally in the models folder
                hf_hub_download(repo_id="wassim249/razi", filename="tf_model.h5", local_dir="../models")
        
        # Load the BERT model weights
        model = TFBertForSequenceClassification.from_pretrained('../models/tf_model.h5', config=config)

        # Load the tokenizer
        tokenizer = BertTokenizer.from_pretrained(tokenizer_folder)
        
        # Load the label encoder
        with open(label_encoder_file, "rb") as f:
            label_encoder = load(f)

        return model , tokenizer , label_encoder
    
    
    @tool
    def get_mentall_illness_category(input_text:str):
        """
        Predict the mental health label of the input text
        params:
            input_text: string
        return:
            probabilities: numpy array
        """
        tokens = bert_tokenizer.encode_plus(
        input_text,
        max_length=128,  # Adjust the max length as needed
        padding="max_length",
        truncation=True,
        return_tensors="tf"
        )
        # Make predictions
        outputs = bert_model(tokens.input_ids, attention_mask=tokens.attention_mask,)
        logits = outputs.logits

        # Apply softmax to obtain probabilities
        probabilities = tf.nn.softmax(logits, axis=1)

        # Convert probabilities to a NumPy array
        probabilities = probabilities.numpy()

        
        #inverse probabilities to get string labels
        labels = label_encoder.inverse_transform( [i for i in range(len(probabilities[0]))])
        probabilities = {label: f'{np.round(prob,2)}%' for label, prob in zip(labels, probabilities[0])}

        return probabilities

    def get_agent(self):
        return self.agent_executor
    



bert_model , bert_tokenizer , label_encoder = RAZIAgent.load_model(config_file="./models/config.json",
                                                                          label_encoder_file="./models/lbl_encoder.pkl",
                                                                          tokenizer_folder="./models/tokenizer")
    

