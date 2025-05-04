Word Embedding Space Analysis with Back-Translation and GPT Augmentation; Evaluate with LSTM binary classifier 

* Overview:
    This project explores how different text augmentation techniques influence word embedding distributions, and how the geometric and     topological structure of the word embedding can tell us about their downstream model performance. 

* Features:
    Back-Translation Augmentation: Text is translated into an intermediate language (Chinese or German) and then translated back to English.

    GPT-Based Paraphrasing: OpenAI’s GPT model generates paraphrased versions of sentences.

    Word Embeddings: Uses Google’s pre-trained Word2Vec (GoogleNews-vectors-negative300) for sentence-level embedding aggregation.

    Geometric & Topological Analysis: Examines structural changes in embedding space.

    Persistent Homology: Captures topological differences in sentence embeddings.

    Matplotlib-Based Visualizations: Generates comparative plots to illustrate differences in embedding spaces across augmentations.

    LSTM Binary Classifier: Trained with Lightning library to track loss and accuracy; applied Sigmoid function to the output of the final                                 layer; used Word2Vec pretrained vector as freezed embedding layer; training loss with BCE. 

* Dataset: 
    I utilize the SST2 sentiment dataset (Stanford Sentiment Treebank 2) as our benchmark dataset, extracting and preprocessing textual data before augmentation.

* Dependencies:
    To run this project, install the required Python libraries:

    pip install numpy matplotlib gensim datasets ripser persim scikit-learn sentence-transformers openai requests
