![User Interface](images/snip.PNG)
The ChatWithPDF RAG (Retrieval-Augmented Generation) application is a web-based tool that allows users to interact with PDF documents by asking questions and receiving detailed answers based on the content within the PDFs. <br>
Users upload PDF files, and the application extracts the text from the documents, splits it into manageable chunks, and stores it in a FAISS vector store.<br>
When users input questions related to the document, the system performs a similarity search to retrieve relevant sections and uses Google’s Gemini model to generate accurate, context-based responses.<br>
This tool provides a conversational interface for users to quickly find specific information within large PDF documents, making it ideal for research, legal, and academic purposes.<br>
It combines natural language processing and retrieval techniques to offer a powerful and efficient question-answering experience.