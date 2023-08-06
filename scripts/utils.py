import tensorflow as tf
import numpy as np
from transformers import BertTokenizer, TFBertForSequenceClassification , BertConfig
from pickle import load
import os
import streamlit as st
from huggingface_hub import hf_hub_download

def build_prompt(user_text,illness):
    """
    Build the prompt for the GPT-3 model
    params:
        user_text: string
        illness: string
    return:
        prompt: string
    """
    return f"""
    A person is diagnosed with {illness} and is experiencing the following symptoms:
    {user_text}
    recommend some tips to help them feel better using this format:
    To feel better you need to :
      1. Tip 1
      2. Tip 2
      ...
    """

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
    if not os.path.exists('./models/tf_model.h5'):

        with st.spinner("Please wait we are downloading the model..."):
            # download the model from the Hugging Face Hub and save it locally in the models folder
            hf_hub_download(repo_id="wassim249/razi", filename="tf_model.h5", local_dir="./models")
        
    # Load the BERT model weights
    model = TFBertForSequenceClassification.from_pretrained('./models/tf_model.h5', config=config)

    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained(tokenizer_folder)

    # Load the label encoder
    with open(label_encoder_file, "rb") as f:
        label_encoder = load(f)

    return model , tokenizer , label_encoder

def predict(bert_model , tokenizer ,lbl_encoder, input_text):
    """
    Predict the label of the input text
    params:
        bert_model: TFBertForSequenceClassification
        tokenizer: BertTokenizer
        lbl_encoder: LabelEncoder
        input_text: string
    return:
        predicted_label: string
        probabilities: numpy array
    """
    tokens = tokenizer.encode_plus(
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

    # Get the predicted label
    predicted_label =lbl_encoder.inverse_transform( [int(np.argmax(probabilities, axis=1)[0])])

    # Print the predicted label and probabilities
    print("Predicted Label:", predicted_label)
    print("Probabilities:", probabilities)

    return predicted_label[0] , probabilities



    
