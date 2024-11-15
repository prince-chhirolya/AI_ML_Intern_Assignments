from langchain.prompts import PromptTemplate
#from langchain.llms import CTransformers
from langchain_community.llms import CTransformers
import json
from transformers import AutoModel

def getLLamaresponse(complaint):
    ### LLama2 model
    llm=CTransformers(model='models\llama-2-7b-chat.ggmlv3.q8_0.bin',
                      model_type='llama',
                      config={'max_new_tokens':256,
                              'temperature':0.01})
    
    ## Prompt Template
    template="""
    You are a customer support officer assigned to analyze the following complaint related to a financial product:

    Complaint: "{complaint}"
    Please analyze this complaint and provide brief, actionable suggestions to improve it. Ensure that:

    The suggestions are exactly two lines long and concise.
    If key details, such as transaction date, account number, or involved parties, are missing, remind the user to include them.
    Your response should be in JSON format with the following fields:

    analysis: A brief assessment of the complaint content and clarity.
    suggestion: Two lines suggesting how the complaint could be improved.
    category: The category of financial issue to which this complaint might belong (e.g., billing, unauthorized charges, service delay).

    Note: If the complaint is incomplete, unclear, or blank, indicate this in the analysis without making assumptions.
    Ensure that your response aligns with the prompt requirements.
    """
    
    prompt=PromptTemplate(input_variables=["complaint"],
                          template=template)
    
    ## Generate the ressponse from the LLama 2 model
    response=llm(prompt.format(complaint=complaint))
    return response

