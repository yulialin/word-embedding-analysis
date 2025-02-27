Word Embedding Space Analysis with Back-Translation and GPT Augmentation

* Overview:
    This project explores how different text augmentation techniques influence word embedding distributions.
    Employed back-translation, GPT augmentation, and persistent homology to analyze the geometric structure of word embeddings derived from the SST2 dataset. Our study utilizes PCA, convex hulls,
    and Delaunay triangulations to visualize transformations in embedding space.

* Features:

    Back-Translation Augmentation: Text is translated into an intermediate language (Chinese or German) and then translated back to English.

    GPT-Based Paraphrasing: OpenAI’s GPT model generates paraphrased versions of sentences.

    Word Embeddings: Uses Google’s pre-trained Word2Vec (GoogleNews-vectors-negative300) for sentence-level embedding aggregation.

    Geometric & Topological Analysis: Convex Hulls & Delaunay Triangulations: Examines structural changes in embedding space.

    Persistent Homology: Captures topological differences in sentence embeddings.

    Matplotlib-Based Visualizations: Generates comparative plots to illustrate differences in embedding spaces across augmentations.

* Dataset: 
    I utilize the GLUE SST2 dataset (Stanford Sentiment Treebank 2) as our benchmark dataset, extracting and preprocessing textual data before augmentation.

* Dependencies:
    To run this project, install the required Python libraries:

    pip install numpy matplotlib gensim datasets ripser persim scikit-learn sentence-transformers openai requests
