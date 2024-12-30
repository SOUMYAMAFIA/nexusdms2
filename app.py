import streamlit as st
import os
import pandas as pd
import json
import requests
import plotly.express as px
import numpy as np
import fitz  # PyMuPDF
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import QueryType

# Azure Document Intelligence Client Setup
endpoint = "https://nexus-docintelligence.cognitiveservices.azure.com/"
key = "5qnO8ROrY6OIYJ0mD6ucbTdHwpfnTfFJVg4CDqkZtwr9SYRy9jhGJQQJ99ALACYeBjFXJ3w3AAALACOG5INJ"
document_analysis_client = DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))

# Azure Blob Storage Client Setup
blob_service_client = BlobServiceClient.from_connection_string(
    "DefaultEndpointsProtocol=https;AccountName=nexusdmscontainer;AccountKey=CEJ/Lop0ZE+McD9kFDo2rOUFfZqAFOyO3oT1WwAu89I8dxru2zXmbq5wsG+1m5CyLEyrxCqXGbrW+AStDlK4Rw==;EndpointSuffix=core.windows.net"
)
container_name = "documents"

# Azure Cognitive Search Client Setup
search_endpoint = "https://nexusdms.search.windows.net"
search_key = "XhOwiPYAqUVeBSe5eWgUJD2JkCNNraxgQrDfGAvvM1AzSeBI5yzf"
index_name = "documents-index"

search_client = SearchClient(endpoint=search_endpoint, index_name=index_name, credential=AzureKeyCredential(search_key))

# Function for Document Upload and Annotation
def document_upload_page():
    st.title("Nexus DMS: Document Upload and Annotation")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        classification, all_responses, all_bboxes, local_file_path = process_pdf(uploaded_file)
        
        st.sidebar.subheader("Document Classification")
        st.sidebar.write(classification)

        st.sidebar.subheader("Extracted Entities (Per Page)")
        for page_num, response in enumerate(all_responses, start=1):
            st.sidebar.write(f"Page {page_num}")
            st.sidebar.json(response.get("entities", {}))

        st.subheader("Annotated Images (Per Page)")
        all_figs = annotate_image_with_plotly(local_file_path, all_bboxes)
        for page_num, fig in enumerate(all_figs, start=1):
            st.write(f"Page {page_num}")
            st.plotly_chart(fig)

# Query Documents Page with Hybrid Search Implementation
def query_documents_page():
    st.title("Nexus DMS: Query Documents")

    query = st.text_input("Enter your query:")
    if query:
        results = perform_hybrid_search(query)

        st.subheader("Search Results")
        for i, result in enumerate(results):
            st.write(f"**Result {i + 1}:**")
            st.write(f"**Score:** {result['@search.score']}")
            st.write(f"**Content:** {result.get('content', 'No content available')}")
            st.json(result)

        # Optionally call GPT-4 to refine or analyze results
        refined_response = refine_query_with_gpt4(query, results)
        st.subheader("GPT-4 Analysis and Insights")
        st.write(refined_response)

# Main App Function
def main():
    logo_path = "nekko logo black bg.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=200)

    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page:", ["Document Upload", "Query Documents"])

    if page == "Document Upload":
        document_upload_page()
    elif page == "Query Documents":
        query_documents_page()

# Azure Cognitive Search Hybrid Search
def perform_hybrid_search(query):
    search_results = search_client.search(
        search_text=query,
        query_type=QueryType.SEMANTIC,
        query_language="en-us",
        semantic_configuration_name="default",
        top=5
    )
    results = [result for result in search_results]
    return results

# Refine Query Results Using GPT-4
def refine_query_with_gpt4(query, search_results):
    formatted_results = json.dumps(search_results, indent=4)
    refined_query = f"""
    Given the following search results for the query "{query}", provide insights, summaries, and recommendations for leadership:
    
    Results:
    {formatted_results}
    """
    return call_gpt4_api(refined_query)

# Helper Functions
# Updated Azure OCR and Bounding Box Extraction
def analyze_pdf(pdf_file):
    with open(pdf_file, "rb") as file:
        poller = document_analysis_client.begin_analyze_document(model_id="prebuilt-read", document=file)
        result = poller.result()
    all_content = []
    all_bboxes = []
    for page in result.pages:
        content = ""
        bboxes = []
        width, height = page.width, page.height
        for line in page.lines:
            content += line.content + "\n"
            bboxes.append([(p.x / width, p.y / height) for p in line.polygon])
        all_content.append(content)
        all_bboxes.append(bboxes)
    return all_content, all_bboxes

def process_pdf(uploaded_file):
    # Reading the uploaded file content
    pdf_bytes = uploaded_file.read()
    pdf_filename = uploaded_file.name
    temp_file_path = os.path.join("tmp", pdf_filename)
    os.makedirs("tmp", exist_ok=True)

    # Save the PDF temporarily on the local file system
    with open(temp_file_path, "wb") as f:
        f.write(pdf_bytes)

    all_content, all_bboxes = analyze_pdf(temp_file_path)
    all_responses = []

    for page_content in all_content:
        response = call_gpt4_api(page_content)
        response = response[7:-3].replace("\n", "").replace("\\n", "")

        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            st.error("Failed to decode JSON response for one of the pages.")
            response = {"category": "Uncategorized", "entities": {}}
        
        all_responses.append(response)

    # Extract the classification for the entire document (use the first page classification as default)
    classification = all_responses[0].get("category", "Uncategorized") if all_responses else "Uncategorized"
    classified_file_path = f"{classification}/{pdf_filename}"

    # Upload and Download Logic
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=classified_file_path)
    try:
        # Upload the PDF to Azure Blob Storage
        with open(temp_file_path, "rb") as f:
            blob_client.upload_blob(f, overwrite=True)
    except Exception as e:
        st.error(f"Error uploading file: {e}")

    local_download_path = os.path.join("tmp", "downloaded_" + pdf_filename)
    try:
        with open(local_download_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
    except Exception as e:
        st.error(f"Error downloading file: {e}")
        return None, None, None, None

    # Clean up the temporary file
    clean_up_local_files(temp_file_path)

    return classification, all_responses, all_bboxes, local_download_path

# Function to clean up temporary files
def clean_up_local_files(*file_paths):
    for file_path in file_paths:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                st.error(f"Error deleting file {file_path}: {e}")

# Annotated Image with Bounding Boxes
def annotate_image_with_plotly(pdf_file_path, all_bboxes):
    doc = fitz.open(pdf_file_path)
    all_figs = []
    for page_num, bboxes in enumerate(all_bboxes):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        fig = px.imshow(img)
        for bbox in bboxes:
            shape = bbox_to_shape(bbox, pix.width, pix.height)
            fig.add_shape(shape)
        all_figs.append(fig)
    return all_figs

# Function to convert a bounding box to a rectangle shape for Plotly
def bbox_to_shape(bbox, width, height):
    x_min = int(bbox[0][0] * width)
    y_min = int(bbox[0][1] * height)
    x_max = int(bbox[2][0] * width)
    y_max = int(bbox[2][1] * height)
    return {
        'type': 'rect',
        'x0': x_min,
        'y0': y_min,
        'x1': x_max,
        'y1': y_max,
        'line': {
            'color': 'rgba(255, 0, 0, 0.8)',
            'width': 2,
        },
    }

# GPT-4 API call function for classification
def call_gpt4_api(prompt):
    url = "https://oainekko.openai.azure.com/openai/deployments/gpt-4o-nekko/chat/completions?api-version=2024-08-01-preview"
    headers = {
        "Content-Type": "application/json",
        "api-key": "cacf6dcb95134cdeab048a36fb6232eb"
    }
    messages = [
        {"role": "system", "content": """You are Nexus DMS, The world's most advanced Document Management System. Your task is to go through the document contents and do the following:
        1. Assign a `category` to the document. The `category` should be assigned based on the contents of the document and should be as descriptive as possible. Example an invoice charging the company for background verification of new joinees could be categorized as `Employee Management/Invoice/Background Verification`.
        2. Extract relevant entities/ necessary fields from the document.  (depending upon the type of document)
        3. Return the result in structured JSON format. (Note: All Values are to be saved as String)"""},  
        {"role": "user", "content": f"Please find the document contents extracted as text: ```{prompt}```"}
    ]
    payload = {
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 4096
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

if __name__ == "__main__":
    main()
