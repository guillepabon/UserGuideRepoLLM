import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import base64
from transformers import AutoTokenizer, AutoModelForCausalLM
from PyPDF2 import PdfReader
from nltk.tokenize import sent_tokenize

# Load the dataset
df = pd.read_csv('user_guides.csv')

# Initialize FAISS index
def create_faiss_index(data):
    embeddings = np.array([get_embedding(text) for text in data])
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

# Get embedding using a pre-trained model
def get_embedding(text, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embedding = model.encode(text, convert_to_tensor=True)
    return embedding.cpu().numpy()

# Search for similar texts
def search(query, index, data, top_k=1):
    query_embedding = get_embedding(query).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    return [data.iloc[i] for i in indices[0]]

# Extract relevant sections from the guide
def extract_relevant_sections(query, guide_text, top_k=5):
    sentences = sent_tokenize(guide_text)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query_embedding = model.encode(query, convert_to_tensor=True)
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    similarities = torch.nn.functional.cosine_similarity(query_embedding, sentence_embeddings)
    top_k_indices = torch.topk(similarities, top_k).indices
    relevant_sentences = [sentences[idx] for idx in top_k_indices]
    return " ".join(relevant_sentences)

# Generate response using a pre-trained model
def generate_response(query, context, model_name="distilgpt2", chunk_size=512):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add a pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))  # Resize model's embedding matrix to accommodate new pad token

    # Extract relevant sections from the context
    relevant_context = extract_relevant_sections(query, context)

    input_text = f"Assistant: Based on the user guides, here is the information:\n{relevant_context}\n"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)

    input_ids = inputs['input_ids'][0]
    max_length = model.config.n_positions

    # Truncate input_ids to the maximum length allowed by the model
    input_ids = input_ids[:max_length]

    # Split input_ids into chunks
    chunks = [input_ids[i:i+chunk_size] for i in range(0, len(input_ids), chunk_size)]

    responses = []
    for chunk in chunks:
        inputs['input_ids'] = chunk.unsqueeze(0)
        outputs = model.generate(inputs['input_ids'], max_new_tokens=150)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        responses.append(response)

    return ' '.join(responses)

# Function to read the full text of the guide from the PDF
def read_pdf_text(file_path):
    text = ""
    with open(file_path, "rb") as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Function to create a download link for a PDF
def create_download_link(file_path, file_name):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}">Download {file_name}</a>'
    return href

# Streamlit UI
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #f4f4f4;
        padding: 10px;
    }
    .stButton>button {
        color: white;
        background-color: #4CAF50;
    }
    .stDownload>button {
        color: white;
        background-color: #4CAF50;
    }
    .icon-text {
        display: flex;
        align-items: center;
    }
    .icon-text img {
        width: 24px;
        height: 24px;
        margin-right: 8px;
    }
    .sidebar .stMarkdown h1, .sidebar .stMarkdown h2, .sidebar .stMarkdown h3, .sidebar .stMarkdown h4, .sidebar .stMarkdown h5, .sidebar .stMarkdown h6 {
        color: #C92043;
    }
    .main-title {
        color: #C92043;
        text-align: center;
    }
    .main-description {
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True
)

# Add logo at the top right corner
st.sidebar.markdown(
    """
    <div style="text-align: center;">
        <img src="https://www.logoai.com/img/home/logoai-new.svg" alt="Logo" style="width: 100px;"/>
    </div>
    """, unsafe_allow_html=True
)

st.sidebar.title("Browse User Guides")
guide_names = df['title'].tolist()
selected_guide = st.sidebar.selectbox("Select a guide to view", guide_names)

if selected_guide:
    guide_info = df[df['title'] == selected_guide].iloc[0]
    guide_path = guide_info['pdf_path']
    guide_text = read_pdf_text(guide_path)

    st.sidebar.write(f"### {selected_guide}")
    st.sidebar.write(guide_info['description'])
    download_link = create_download_link(guide_path, f'{guide_info["title"]}.pdf')
    st.sidebar.markdown(download_link, unsafe_allow_html=True)
    st.sidebar.write(guide_text[:200] + "...")  # Show a preview of the guide content

# Icons and visual enhancements using HTML and CSS
st.sidebar.markdown('<div class="icon-text"><img src="https://cdn.icon-icons.com/icons2/935/PNG/512/menu-square-button_icon-icons.com_73216.png"/> Browse Guides</div>', unsafe_allow_html=True)
for guide in guide_names:
    st.sidebar.markdown(f'<div class="icon-text"><img src="https://cdn1.iconfinder.com/data/icons/flat-design-basic-set-4/24/document-blue-text-portrait-1024.png"/> {guide}</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="icon-text"><img src="https://cdn4.iconfinder.com/data/icons/evil-icons-user-interface/64/question-1024.png"/> Ask a Question</div>', unsafe_allow_html=True)

st.markdown("<h1 class='main-title'> User Guide Repository Query System</h1>", unsafe_allow_html=True)
st.markdown("<p class='main-description'>Ask a question about the user guides to find relevant products and/or specific operations.</p>", unsafe_allow_html=True)

query = st.text_input("Enter your query here, make sure to be as clear as possible:")
if query:
    index = create_faiss_index(df['text'])
    results = search(query, index, df)

    # Select the most relevant document and load its full text
    if results:
        result = results[0]
        pdf_path = result['pdf_path']
        guide_text = read_pdf_text(pdf_path)
        context = guide_text
        response = generate_response(query, context)

        st.write("### Response:")
        st.write(response)

        # Display the matched document
        st.write("### Matched Document:")
        st.write(result['text'])
        download_link = create_download_link(result['pdf_path'], f'{result["title"]}.pdf')
        st.markdown(download_link, unsafe_allow_html=True)
    else:
        st.write("No relevant documents found.")