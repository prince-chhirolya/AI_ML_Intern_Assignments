import streamlit as st
from src.llama import getLLamaresponse,getLLamaAPIresponse
from  src.load_lstm import LSTM_functions
import re
import json
import pandas as pd
import os

st.set_page_config(page_title="Finaincial Products Complaint",
                    page_icon='🤖',
                    layout='centered',
                    initial_sidebar_state='collapsed')

st.header("Finaincial Product Complaint Reporting🤖")

name=st.text_input("Enter Your Name ")
input_text=st.text_area("Enter the complaint")

submit=st.button("Generate")

    ## Final response
if submit:
    answer=getLLamaAPIresponse(input_text)
    print(answer)
    match = re.search(r'\{(.*)\}', answer, re.DOTALL)
    # Extract the matched content and add the curly braces back
    extracted_info = '{' + match.group(1).strip() + '}' if match else None
    extracted_info=json.loads(extracted_info)
    st.write(extracted_info)
    lstm=LSTM_functions()
    preds=lstm.load_and_predict(input_text)
    st.success(preds)

