
## IMPORTING PACKAGES
import os 
import shutil
from openai import AzureOpenAI
from flask import Flask, request, render_template_string
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
import numpy as np
from pdf2image import convert_from_path
import cv2
import pytesseract
import base64
from dotenv import load_dotenv

## CREDENTIALS
load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

print(AZURE_OPENAI_ENDPOINT)
print(AZURE_OPENAI_API_KEY)
print(DEPLOYMENT_NAME)

app = Flask(__name__)

# Load the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Global variable to store the PDF contents
pdf_contents = []
table_contents = []
image_contents = []

UPLOAD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Upload PDF Files</title>
</head>
<body>
    <h1>Upload PDF Files</h1>
    <form action="/uploader" method="post" enctype="multipart/form-data">
        <input type="file" name="files" multiple />
        <input type="submit" />
    </form>
</body>
</html>
"""

RESULT_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>PDF Content</title>
</head>
<body>
    <h1>Extracted PDF Content</h1>
    {% for content in contents %}
    <h2>File {{ loop.index }}</h2>
    <pre>{{ content }}</pre>
    {% endfor %}

    <h1>Extracted Image Texts</h1>
    {% for text in image_contents %}
    <h2>text {{ loop.index }}</h2>
    <pre>{{ text }}</pre>
    {% endfor %}

    <h2>Ask a Question</h2>
    <form action="/ask" method="post">
        <input type="hidden" name="contents" value="{{ contents }}">
        <input type="hidden" name="image_texts" value="{{ image_texts }}">
        <label for="question">Question:</label>
        <input type="text" id="question" name="question" required>
        <input type="submit" value="Ask">
    </form>

    <h2>Summarize PDF</h2>
    <form action="/summarize" method="post">
        <input type="hidden" name="contents" value="{{ contents }}">
        <input type="hidden" name="image_contents" value="{{ image_contents }}">
        <input type="submit" value="Summarize">
    </form>

    {% if answer %}
    <h2>Answer</h2>
    <p><strong>Question:</strong> {{ question }}</p>
    <p><strong>Answer:</strong> {{ answer }}</p>
    {% endif %}

    {% if summary %}
    <h2>Summary</h2>
    <p>{{ summary }}</p>
    {% endif %}
</body>
</html>
"""
temp_image_dir = "./tmp/pdf_uploads/image"
temp_text_dir = "./tmp/pdf_uploads/text"

@app.route('/')
def upload_file():
    return render_template_string(UPLOAD_HTML)

@app.route('/uploader', methods=['GET', 'POST'])
def uploader_file():
    global pdf_contents, image_contents
    if request.method == 'POST':
        files = request.files.getlist('files')
        
        for file in files:
            print(file.filename)
            shutil.rmtree(temp_text_dir)
            os.makedirs(temp_text_dir, exist_ok=True)
            temp_file_path = os.path.join(temp_text_dir, file.filename)
            file.save(temp_file_path)
            
            shutil.rmtree(temp_image_dir)
            os.makedirs(temp_image_dir, exist_ok=True)
            temp_image_path = os.path.join(temp_image_dir, file.filename)
            print(f"cp {temp_file_path} {temp_image_path}")
            os.system(f"cp {temp_file_path} {temp_image_path}")
           
        pdf_contents = [extract_pdf_content(file) for file in os.listdir(temp_text_dir)]
        image_contents = [process_images_from_pdf(file) for file in os.listdir(temp_image_dir)]
        #print(f"pdf_contents: {pdf_contents}")
        #print(f"image_contents: {image_contents[0]}")
            
        return render_template_string(RESULT_HTML, contents=pdf_contents, image_contents=image_contents)

@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.form['question']
            
    nearest_chunk = find_nearest_chunk(question, pdf_contents[0])
    nearest_page = nearest_chunk['page']
    nearest_image = image_contents[0][nearest_page]
    answer = get_answer_from_gpt(question, nearest_chunk, nearest_image)
    return render_template_string(RESULT_HTML, answer=answer, question=question)

@app.route('/summarize', methods=['POST'])
def summarize_pdf():
    combined_text = chain_all_text(pdf_contents[0])   
    print(f"combined_text: {combined_text}") 
        
    summary = get_summary_from_gpt(combined_text)
    return render_template_string(RESULT_HTML, contents=[combined_text], summary=summary)

def merge_lists(list1, list2):
    merged_dict = {}

    # Process the first list
    for item in list1:
        if isinstance(item, dict) and 'page' in item:
            page = item['page']
            if page not in merged_dict:
                merged_dict[page] = {'page': page, 'text': item['text']}
            else:
                merged_dict[page]['text'] += item['text']

    # Process the second list
    for item in list2:
        if isinstance(item, dict) and 'page' in item:
            page = item['page']
            if page not in merged_dict:
                merged_dict[page] = {'page': page, 'text': item['text']}
                if 'image' in item:
                    merged_dict[page]['image'] = item['image']
            else:
                merged_dict[page]['text'] += item['text']
                if 'image' in item:
                    merged_dict[page]['image'] = item['image']

    # Convert the merged dictionary back to a list
    merged_list = list(merged_dict.values())
    return merged_list

def chain_all_text(merged_list):
    all_text = ""
    for item in merged_list:
        all_text += item['text'] + "\n\n"
    return all_text.strip()

def extract_pdf_content(file):
    # Save the uploaded file to a temporary location
    temp_file_path = os.path.join(temp_text_dir, file)
    doc = fitz.open(temp_file_path, filetype="pdf")
    
    contents = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
            
        tab_content = ""
        tabs = page.find_tables()
        for i,tab in enumerate(tabs):
            tab_content += "\n\n" + tab.to_pandas().to_string()
           
        contents.append({
            "page": page_num,
            "text": text,
            "table": tab_content
        })
    ''' 
    for content in contents:
        for key, value in content.items():
            print(f"{key}:{value}")
    '''        
    return contents

def process_images_from_pdf(file):
    # Save the uploaded file to a temporary location
    temp_file_path = os.path.join(temp_image_dir, file)
    print(temp_file_path)
    
    # Convert PDF to images
    images = convert_from_path(temp_file_path)
    contents = []
    for page_num in range(len(images)):
        image = images[page_num]
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray, fx=0.5, fy=0.5, dsize=(0, 0))
        cv2.imwrite(os.path.join(temp_image_dir,f"{file}_{page_num}.png"), resized_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        with open(os.path.join(temp_image_dir,f"{file}_{page_num}.png"), "rb") as image_file:
            base64_str = base64.b64encode(image_file.read()).decode('utf-8')
    
        # Apply OCR to extract text
        text = pytesseract.image_to_string(gray)
        contents.append({
            "page": page_num,
            "text":text,
            "image": base64_str
        })
    '''
    for content in contents:
        for key, value in content.items():
            print(f"{key}:{value}")
    '''        
    return contents


# Example function to find the nearest chunk
def find_nearest_chunk(query, chunks):
    # Implement your similarity calculation here
    question_embedding = model.encode(query, convert_to_tensor=True)
    nearest_chunk = max(chunks, key=lambda chunk: util.pytorch_cos_sim(question_embedding, model.encode(chunk["text"], convert_to_tensor=True).flatten()))
    print(f"nearest_chunk: {nearest_chunk}")
    return nearest_chunk


def retrieve_relevant_text(content, question):
    # Split the content into chunks (e.g., paragraphs)
    chunks = content
    
    # Encode the chunks and the question using the Sentence Transformer model
    chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
    question_embedding = model.encode(question, convert_to_tensor=True)
    
    # Compute cosine similarity between the question and each chunk
    cosine_similarities = util.pytorch_cos_sim(question_embedding, chunk_embeddings).flatten()
    
    # Get the most relevant chunk
    most_relevant_index = np.argmax(cosine_similarities)
    relevant_text = chunks[most_relevant_index]
    
    return relevant_text

def get_answer_from_gpt(question, nearest_chunk, nearest_image):
    print(f"get_answer_from_gpt - nearest_chunk: {nearest_chunk}")
    print(f"get_answer_from_gpt - nearest_image: {nearest_image}")
    relevant_text = nearest_chunk["text"]
    image_encoded = nearest_image["image"]
    endpoint = AZURE_OPENAI_ENDPOINT
    deployment = DEPLOYMENT_NAME
    api_key=AZURE_OPENAI_API_KEY
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version="2024-02-01",
    )
    completion = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": f'''You are an electronic engineer. You understand that scientific chart the user is referring to you. 
                The axises in these charts may be in linear or log_10 scale.
                The axes may be labeled with various units, such as Hz, kHz, MHz, GHz, or THz for frequency, and dB, dBm, or dBW for power. 
                The axes may also not start from zero. 
                If the user is asking you a question about the chart. You need to provide the answer based on the chart.'''},
            {"role": "user", "content": f"Based on the following PDF content, answer the question:\n\nPDF Content:\ndata:image/png;base64,{image_encoded}\n{relevant_text}\n\nQuestion: {question}\n\nAnswer:"}
        ],
        max_tokens=200,
        temperature=0.8
    )
    
    return completion.choices[0].message.content.strip()

def get_summary_from_gpt(content):
    endpoint = AZURE_OPENAI_ENDPOINT
    deployment = DEPLOYMENT_NAME
    api_key=AZURE_OPENAI_API_KEY
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version="2024-02-01",
    )
    completion = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Summarize the following PDF content:\n\nPDF Content:\n{content}\n\nSummary:"}
        ],
        max_tokens=200, 
        temperature=0.0
    )
    return completion.choices[0].message.content.strip()

if __name__ == '__main__':
    app.run(debug=True)    