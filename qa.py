from pytube import YouTube
import whisper
import pandas as pd
import math
import tiktoken
import os
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import streamlit as st
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

def get_pipeline():

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-tiny-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    return pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
    )


def get_video_transcription(video_url):
    audio_file = YouTube(video_url).streams.filter(only_audio=True).first().download(filename="audio.mp4")

    pipe = get_pipeline()

    transcription = pipe(audio_file)

    return transcription["chunks"]



def get_text_chunks_metadata(video_url,chunk_size=100):
    chunks = get_video_transcription(video_url)

    texts = []
    metadata = []
    size = len(chunks)
    i = 0

    while i < size:
        text = ""
        chunk_size = 100
        chunk_counter = 0
        start = math.floor(chunks[i]["timestamp"][0])
        end = math.floor(chunks[i]["timestamp"][1])
        while i < size and chunk_counter <= chunk_size:
            t =  chunks[i]["text"]
            text = text+" "+t
            end = math.floor(chunks[i]["timestamp"][1])
            chunk_counter += len(t)
            i+=1
        texts.append(text)
        metadata.append({"start":start,"end":end})   
    
    return texts, metadata


def load_pinecone(video_url,chunk_size=100):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    # initialize pinecone
    pinecone.init(
    api_key=st.secrets['PINECONE_API_KEY'],
    environment=st.secrets['PINECONE_ENV']
    )
    index_name = st.secrets['PINECONE_NAME']

    if index_name not in pinecone.list_indexes():
         # we create a new index
        pinecone.create_index(
            name=index_name,
            dimension=1536  
        )
    else:
        return Pinecone.from_existing_index(index_name,embeddings)

    texts, metadata = get_text_chunks_metadata(video_url,chunk_size)
    docs = []
    for i in range(len(texts)):
        doc = Document(
        page_content=texts[i],
        metadata=metadata[i]
        )
        docs.append(doc)

    return Pinecone.from_documents(docs, embeddings, index_name=index_name)    

def get_similiar_docs(query,index,k=2, score=False):
    if score:
        similar_docs = index.similarity_search_with_score(query, k=k)
    else:
        similar_docs = index.similarity_search(query, k=k)
    return similar_docs


def get_index():
    return load_pinecone("",400)

def qa_answer(query):
    index = get_index()

    model_name = "gpt-3.5-turbo"
    llm = OpenAI(model_name=model_name)
    chain = load_qa_chain(llm, chain_type="stuff")

    similar_docs_1 = get_similiar_docs(query,index,4,False)

    similar_docs_2 = get_similiar_docs(query,index,4,True)

    doc_start = similar_docs_2[0][0].metadata

    doc_end = similar_docs_2[3][0].metadata

    start = math.floor(doc_start["start"])

    end = math.floor(doc_end["end"])

    answer = chain.run(input_documents= similar_docs_1 , question=query)

    value = answer + "\n" + f"You can watch it from start {start} to {end}"

    return value, start, end





