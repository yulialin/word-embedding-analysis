import os
import json
import re
import uuid
import requests
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from datasets import load_dataset
from ripser import ripser
import persim
from persim import sliced_wasserstein
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from scipy.spatial import ConvexHull, Delaunay
import math

# Set up API keys and endpoints
os.environ['OPENAI_API_KEY'] = "[replace with key]"
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

# Azure Translator API settings
AZURE_SUBSCRIPTION_KEY = "[replace with key]"
AZURE_ENDPOINT = "https://api.cognitive.microsofttranslator.com"
AZURE_LOCATION = "eastus"

def translate_text(text, from_language="en", to_languages=["fr"]):
    """
    Translates text using the Azure Translator API.
    """
    path = '/translate?api-version=3.0'
    url = AZURE_ENDPOINT + path

    headers = {
        'Ocp-Apim-Subscription-Key': AZURE_SUBSCRIPTION_KEY,
        'Ocp-Apim-Subscription-Region': AZURE_LOCATION,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }

    params = {
        'from': from_language,
        'to': to_languages
    }

    body = [{'text': text}]
    response = requests.post(url, headers=headers, params=params, json=body)

    if response.status_code != 200:
        print(f"Error: {response.status_code}, {response.text}")
        raise Exception(f"API call failed with status code {response.status_code}")

    return response.json()

def back_translate(text, intermediate_language="fr"):
    """
    Performs back-translation: English -> Intermediate Language -> English.
    """
    # Translate to the intermediate language
    intermediate_result = translate_text(text, to_languages=[intermediate_language])
    translated_text = intermediate_result[0]['translations'][0]['text']
    # Translate back to English
    back_result = translate_text(translated_text, from_language=intermediate_language, to_languages=["en"])
    return back_result[0]['translations'][0]['text']


# GPT Augmentation Function
def gpt_augment_text(text, max_tokens=100, temperature=0.7):
    prompt = (
        "Paraphrase the following sentence, preserving its meaning. "
        "Output only the paraphrased sentence, with no extra commentary or formatting:\n\n"
        f"Sentence: {text}\nParaphrase:"
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that paraphrases text."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    # parse if neccessary
    content = response.choices[0].message.content.strip()
    return content


# Data Cleaning Function
def clean_text(sentence):
    """
    Cleans the text by converting to lowercase and replacing punctuation with spaces.
    """
    sentence = sentence.lower()
    sentence = re.sub(r'[^\w\s]', ' ', sentence)  # Replace punctuation with space
    return re.sub(r'\s+', ' ', sentence).strip()


# Sentence-Level Embedding
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
def sentence_embeddings(sentences):
    return sentence_model.encode(sentences, convert_to_tensor=False)

# Word Embeddings for Individual Words using Word2Vec
# adpating the og paper style, gathering all sentence-level arrays into one big array
word_embedding_cache = {}
def get_word_vectors(sentence, model):
    word_vectors = []
    words = sentence.split()
    for word in words:
        if word in word_embedding_cache:
            word_vectors.append(word_embedding_cache[word])
        elif word in model:
            word_vectors.append(model[word])
            word_embedding_cache[word] = model[word]
        else:
            print(f"Word '{word}' not found in the Word2Vec vocabulary.")
    return np.array(word_vectors)

def gather_word_vectors(sentences, model):
    """
    Takes a list of sentences, calls get_word_vectors for each,
    and returns a single (total_words, embedding_dim) array.
    """
    all_vecs = []
    for sent in sentences:
        vecs = get_word_vectors(sent, model)  # your existing function
        # Only append if the sentence wasn't empty or OOV
        if len(vecs) > 0:
            all_vecs.append(vecs)
    if len(all_vecs) == 0:
        return np.array([]).reshape(0, 300)  # empty
    return np.vstack(all_vecs)  # shape (N, 300)


# Persistent Homology and Visualization
def persistent_homology_analysis(vectors, title):
    diagrams = ripser(vectors)['dgms']
    plt.figure()
    persim.plot_diagrams(diagrams, title=title)
    return diagrams

def plot_convex_hull(points, color):
    if len(points) >= 3:
        hull = ConvexHull(points)
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], color=color, linewidth=2)

def plot_delaunay(points, color='gray'):
    if len(points) >= 3:
        delaunay = Delaunay(points)
        for simplex in delaunay.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], color=color, linewidth=1, alpha=0.7)

def count_h1_features(vectors):
    diagrams = ripser(vectors)['dgms']
    h1_features = len(diagrams[1])
    return h1_features

def compute_convex_hull_stats(points):
     """
    Computes statistics for the convex hull of a set of 2D points.
    
    Parameters:
        points (ndarray): A 2D numpy array of shape (N, 2) representing N points.
    
    Returns:
        dict: Contains convex hull area and perimeter.
    """
     hull = ConvexHull(points)
     area = hull.volume
     

if __name__ == "__main__":
    # Load or create cache
    cache_file = "augmented_cache.json"
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_dict = json.load(f)
    else:
        cache_dict = {}
      
    # Load data - using only the first 20 examples
    dataset = load_dataset("glue", "sst2", split="train[:11855]")
    raw_sentences = [item["sentence"] for item in dataset]
    all_sentences = [clean_text(s) for s in raw_sentences]

    # Process each sentence with skip logic
    for i, sentence in enumerate(all_sentences):
        if sentence in cache_dict:
            continue
        
        print(f"Processing line {i}: {sentence[:60]}...")
        gpt_text = gpt_augment_text(sentence)
        bt_zh = back_translate(sentence, intermediate_language="zh-Hans")
        bt_de = back_translate(sentence, intermediate_language="de")

        cache_dict[sentence] = {
            "gpt": gpt_text,
            "bt_zh": bt_zh,
            "bt_de": bt_de
        }

        if i % 50 == 0:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_dict, f, ensure_ascii=False, indent=4)
            print(f"Partial save at line {i}")

    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache_dict, f, ensure_ascii=False, indent=4)
    print("All sentences processed and saved to augmented_cache.json")

    # ***** Filter the cache to use only the current dataset entries *****
    filtered_original_texts = []
    filtered_gpt_texts = []
    filtered_bt_zh_texts = []
    filtered_bt_de_texts = []

    for sentence in all_sentences:
        if sentence in cache_dict:
            filtered_original_texts.append(sentence)
            filtered_gpt_texts.append(cache_dict[sentence]["gpt"])
            filtered_bt_zh_texts.append(cache_dict[sentence]["bt_zh"])
            filtered_bt_de_texts.append(cache_dict[sentence]["bt_de"])

    # Compute embeddings using the filtered lists
    # Load Word2Vec Model
    print("Loading Word2Vec model (GoogleNews-vectors-negative300.bin)...")
    word2vec = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    emb_original = gather_word_vectors(filtered_original_texts, word2vec)
    emb_gpt = gather_word_vectors(filtered_gpt_texts, word2vec)
    emb_bt_zh = gather_word_vectors(filtered_bt_zh_texts, word2vec)
    emb_bt_de = gather_word_vectors(filtered_bt_de_texts, word2vec)

    # Visualization: PCA and Convex Hull Plots
    n_orig = emb_original.shape[0]
    n_gpt = emb_gpt.shape[0]
    n_bt_de = emb_bt_de.shape[0]
    n_bt_zh = emb_bt_zh.shape[0]
    
    # One PCA for ALL data
    all_data = np.concatenate([emb_original, emb_gpt, emb_bt_de, emb_bt_zh], axis=0)
    pca = PCA(n_components=2)
    all_data_2d = pca.fit_transform(all_data)
    
    # Slice out each group
    orig_2d = all_data_2d[:n_orig]
    gpt_2d = all_data_2d[n_orig:n_orig + n_gpt]
    bt_zh_2d = all_data_2d[n_orig + n_gpt:n_orig + n_gpt + n_bt_zh]
    bt_de_2d = all_data_2d[n_orig + n_gpt + n_bt_zh:n_orig + n_gpt + n_bt_zh + n_bt_de]

    # Plot 1: Original vs. Back-Translated (Chinese)
    plt.figure(figsize=(10, 8))
    plt.scatter(orig_2d[:, 0], orig_2d[:, 1], c='blue', marker='o', label='Original')
    plt.scatter(bt_zh_2d[:, 0], bt_zh_2d[:, 1], c='orange', marker='s', label='Back-Translated (Chinese)')
    plot_delaunay(orig_2d, color='blue')
    plot_delaunay(bt_zh_2d, color='orange')
    plot_convex_hull(orig_2d, 'blue')
    plot_convex_hull(bt_zh_2d, 'orange')
    plt.title("Word Embeddings (Single PCA): Original vs Back-Translated (Chinese)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()

    # Plot 2: Original vs. Back-Translated (German)
    plt.figure(figsize=(10, 8))
    plt.scatter(orig_2d[:, 0], orig_2d[:, 1], c='blue', marker='o', label='Original')
    plt.scatter(bt_de_2d[:, 0], bt_de_2d[:, 1], c='red', marker='^', label='Back-Translated (German)')
    plot_delaunay(orig_2d, color='blue')
    plot_delaunay(bt_de_2d, color='red')
    plot_convex_hull(orig_2d, 'blue')
    plot_convex_hull(bt_de_2d, 'red')
    plt.title("Word Embeddings (Single PCA): Original vs Back-Translated (German)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()

    # Plot 3: Original vs. GPT-Augmented
    plt.figure(figsize=(10, 8))
    plt.scatter(orig_2d[:, 0], orig_2d[:, 1], c='blue', marker='o', label='Original')
    plt.scatter(gpt_2d[:, 0], gpt_2d[:, 1], c='green', marker='D', label='GPT-Augmented')
    plot_delaunay(orig_2d, color='blue')
    plot_delaunay(gpt_2d, color='green')
    plot_convex_hull(orig_2d, 'blue')
    plot_convex_hull(gpt_2d, 'green')
    plt.title("Word Embeddings (Single PCA): Original vs GPT")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()

    # Persistent Homology (optional)
    persistent_homology_analysis(emb_original, "PH: Original Words")
    persistent_homology_analysis(emb_bt_zh, "PH: BT Chinese Words")
    persistent_homology_analysis(emb_bt_de, "PH: BT German Words")
    persistent_homology_analysis(emb_gpt, "PH: GPT Words")

    # Show all plots
    plt.show()
    print("Done. Close all windows to exit.")

