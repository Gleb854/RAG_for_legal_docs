This report details the development of a Retrieval Augmented Generation (RAG) model to enhance the accessibility and comprehension of civil law documents in the Russian Federation. 
By leveraging cutting-edge NLP techniques, the project aims to make legal information more accessible and understandable, promoting transparency and informed citizenship.

MOTIVATION:
The complexity of Russian civil law documents significantly impedes the ability of non-experts to access and understand crucial legal information. This barrier not only undermines transparency and justice but also impacts informed citizenship. Developing a Retrieval-Augmented Generation (RAG) model specifically tailored to Russian legal documents can bridge this gap. By simplifying the legal research process, the RAG model aims to make legal knowledge more accessible not only to the general public but also to legal professionals and businesses, thereby enhancing societal legal empowerment.

DATA PREPROCESS:
The data was carefully parsed, structured, and annotated to train the RAG model effectively, ensuring the model's output is both accurate and legally sound. For our dataset we used different normative legal act of the Russian:

- Гражданский кодекс;
- Гражданский Процессуальный кодекс;
- Жилищный кодекс;
- Налоговый кодекс;
- Земельный кодекс;
- Трудовой кодекс;
- ФЗ

The preprocessed dataset was parsed and split on articles and it's metadata (the name of the article, section, subsection, etc) by the hierarchy of the document. Splitting the data was performed with 100 size chunks with 30 elements overlapping and collected text of the article and metadata for the unit massive for each one. Using RuBert model the embedding was done and with the FAISS (Facebook AI Similarity Search) five the most close vectores are chosen for fine-tuning. The example of the query and the correlated articles based on the FAISS:

Question: Может ли Частное учреждение осуществлять приносящую доходы деятельность?

Query: Гражданский кодекс Статья 303, Гражданский кодекс Статья 228, Налоговый кодекс Статья 275.1, Гражданский кодекс Статья 635, ФЗ ООО Статья 15

MODEL ARCHITECTURE:
The RAG model integrates a retrieval mechanism with a generative NLP model to provide accurate and contextually appropriate legal information. More classical approaches propose to use an End2End architecture that updates both generator and retriever training. Due to some computational limitations, we used a somewhat simpler architecture where the whole retriever is frozen. More precisely, in our project we compared two tuning pipelines: with and without additional information retrieval.
We compared two fine-tuning pipelines: with and without additional information retrieval. In the first setup (without retrieval), we simply fine-tune the phi-3b model on a question-answering dataset. In the second setup, we augment the training question-answer pairs with a top-5 retrieved document from our FAISS index.
