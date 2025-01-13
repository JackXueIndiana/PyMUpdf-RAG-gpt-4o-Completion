# PyMUpdf-RAG-gpt-4o-Completion
With a flask app, we demonstrate how to use library pymupdf to extract text from page, table and image in PDF file and resize the images in pages and associate the image back for LLM completion.

Once a PDF file loaded in, we turn it into a set of pages. For each page, we extract text from the page and tables in the page. We also extract text from the images in the page and reduce the image to a manageable size for late use. The textual contents are then embedded and vectorized. 

We call gpt-40 for summarize the document with the textual content from all pages. For answering specific questions, we find the most proper page based on the cosine similarity calculation between the question vector and the vector represent the pages. Once the page selected, we call the gpt-4o with the textual content and related images in based64 as data:image/png;base64,{image_encoded}.

The following libraries are used:
- Library [pymupdf](https://pypi.org/project/PyMuPDF/) is a high-performance Python library for data extraction, analysis, conversion & manipulation of PDF (and other) documents. 
- [pdf2image](https://pypi.org/project/pdf2image/) wraps pdftoppm and pdftocairo to convert PDF to a PIL Image object, for which we can extract the text in images with library [tesseract](https://pypi.org/project/pytesseract/), an optical character recognition (OCR) tool for python that recognizes and reads the text embedded in images. 
- To gray and resize images we rely on OpenCV's [cv2](https://pypi.org/project/opencv-python/).
-  [Sentence Transformers](https://www.sbert.net/) (a.k.a. SBERT) is the go-to Python module for accessing, using, and training state-of-the-art text and image embedding models.




