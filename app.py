
from langchain.llms import OpenAI
from langchain.agents import create_pandas_dataframe_agent
from langchain.agents import create_csv_agent
import pandas as pd
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
import gradio as gr
import os

import whisper
from audio_recorder_streamlit import audio_recorder
from audiorecorder import audiorecorder

from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.llms.openai import OpenAI
from gtts import gTTS



def call_openai_api(prompt):
    max_retries = 3
    retry_delay = 3600  # seconds

    for _ in range(max_retries):
        try:
            response = openai.Completion.create(
                engine="davinci",
                prompt=prompt,
                max_tokens=10000
            )
            return response.choices[0].text.strip()
        except openai.error.RateLimitError:
            print("Rate limit exceeded. Retrying after delay...")
            time.sleep(retry_delay)
    
    return "Failed to get a response after multiple retries."
# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = 'YOUR API KEY'

# Create the LangChain agent using the OpenAI agent and DataFrame
agent = create_csv_agent(
    OpenAI(temperature=0),
    "predictions.csv",
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)
#the python agent is made if we need some python code 
#agent = create_python_agent(llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),tool=PythonREPLTool(),verbose=True,agent_type=AgentType.OPENAI_FUNCTIONS,agent_executor_kwargs={"handle_parsing_errors": True},)


model = whisper.load_model("base")
model.device
def transcribe(audio):

    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)


    options = whisper.DecodingOptions(fp16 = False)
    result = whisper.decode(model, mel, options)
    result_text = result.text
    
    out_result = agent.run(result_text)
    
    audioobj = gTTS(text = out_result, 
                    slow = False)
    
    audioobj.save("audio.mp3")

    return [result_text, out_result, "audio.mp3"]

output_1 = gr.Textbox(label="Speech to Text")
output_2 = gr.Textbox(label="TradingGPT Output")

output_4="text"

# Define the Gradio interface

iface1 =gr.Interface(
    title="GeoPoliGpt", 
    fn=transcribe, 
    inputs=[
        gr.inputs.Audio(source="microphone", type="filepath")
    ],

    outputs=[
        output_1,  output_2
    ],
    live=True)

iface2 = gr.Interface(
    fn=agent.run,
    inputs=gr.inputs.Textbox(label="Question"),
    outputs=output_4,
    title="GeoPoliGpt"
)

gr.TabbedInterface(
    [iface2, iface1], ["Text Input", "Voice Input"]
).launch(share=True)

