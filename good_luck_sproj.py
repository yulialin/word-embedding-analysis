import os
import json
import re
import uuid
import requests
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from datasets import load_dataset
from ripser import ripser
import persim
from persim import sliced_wasserstein
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from scipy.spatial import ConvexHull, Delaunay
from scipy.stats import wasserstein_distance 
import math
from shapely.geometry import Polygon
import pickle
import ot 
import string

nltk.download('punkt')
nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))

# Set up API keys and endpoints
os.environ['OPENAI_API_KEY'] = "[REPLACE WITH OPENAI API KEY]"
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

# Azure Translator API settings
AZURE_SUBSCRIPTION_KEY = "[REPLACE WITH AZURE API KEY]"
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


# Data Cleaning Function using nltk
def nltk_clean_text(text):
    if text is None:
        return ""
    # Tokenize text into sentences
    sentences = sent_tokenize(text)
    cleaned_sentences = []
    
    for sentence in sentences:
        # Tokenize sentence into words
        words = word_tokenize(sentence)
        cleaned_words = []
        for word in words:
            # Remove all punctuation characters using regex.
            cleaned_word = re.sub(r'[^\w\s]', '', word)
            cleaned_word = cleaned_word.lower()
            # Filter out empty tokens and stopwords
            if cleaned_word and cleaned_word not in stop_words:
                cleaned_words.append(cleaned_word)
        # Only add non-empty sentences
        if cleaned_words:
            cleaned_sentences.append(" ".join(cleaned_words))
    
    # Return the joined cleaned sentences (or an empty string if none)
    return " ".join(cleaned_sentences) if cleaned_sentences else ""


# Sentence-Level Embedding
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
def sentence_embeddings(sentences):
    return sentence_model.encode(sentences, convert_to_tensor=False)

# Word Embeddings for Individual Words using Word2Vec
# adpating the og paper style, gathering all sentence-level arrays into one big array
# also cache for out-of-vobulary words that doesn't have a word vecotor 
word_embedding_cache = {}
oov_words_set = set()

def get_word_vectors(sentence, model):
    word_vectors = []
    words = sentence.split()
    oov_count = 0
    for word in words:
        if word in word_embedding_cache:
            word_vectors.append(word_embedding_cache[word])
        elif word in model:
            vec = model[word]
            word_embedding_cache[word] = vec
            word_vectors.append(vec)
        else:
            oov_count += 1
            oov_words_set.add(word)
            print(f"Word '{word}' not found in the Word2Vec vocabulary.")
    return np.array(word_vectors), oov_count, len(words)

def gather_word_vectors(sentences, model):
    """
    Takes a list of sentences, calls get_word_vectors for each,
    and returns a single (total_words, embedding_dim) array.
    """
    all_vecs = []
    oov_info = [] # list of (oov_count, total_words) for each sentence
    valid_sentences = [] # sentecnes that pass the OOV threshold
    for sent in sentences:
        vecs, oov_count, total_words = get_word_vectors(sent, model)  # your existing function
        # Only append if the sentence wasn't empty or OOV
        oov_info.append((oov_count, total_words))
        if vecs.size > 0 and (oov_count / total_words) <= 0.9: # threshold set to 90%
            all_vecs.append(vecs)
            valid_sentences.append(sent)
        else: 
            print(f"Dropping sentence due to high OOV rate:{sent}")
    if len(all_vecs) == 0:
        return np.array([]).reshape(0, 300), oov_info, valid_sentences  # empty
    return np.vstack(all_vecs), oov_info, valid_sentences # shape (N, 300)

def compute_multivariate_wasserstein(X, Y):
    """
    Compute the 2D Wasserstein (Earth Mover's) distance between two sets of points using POT.
    
    Parameters:
        X (np.ndarray): Array of shape (n_samples_X, 2) for group X.
        Y (np.ndarray): Array of shape (n_samples_Y, 2) for group Y.
        
    Returns:
        float: The 2D Wasserstein distance between the two distributions.
    """
    # Compute cost matrix (Euclidean distances)
    M = ot.dist(X, Y)
    # Normalize the masses
    a = np.ones(X.shape[0]) / X.shape[0]
    b = np.ones(Y.shape[0]) / Y.shape[0]
    # Compute squared Wasserstein distance
    distance2 = ot.emd2(a, b, M)
    return np.sqrt(distance2)

# Persistent Homology and Visualization
def persistent_homology_analysis(vectors, title, visualize=True):
    diagrams = ripser(vectors)['dgms']
    if visualize:
        plt.figure()
        persim.plot_diagrams(diagrams, title=title)

    h0_features = diagrams[0]
    h1_features = diagrams[1]
    print(f"{title} - H0 features count: {len(h0_features)}")
    print(f"{title} - H1 features count: {len(h1_features)}")
    
    # Optionally, print lifetimes of H1 features
    lifetimes = h1_features[:, 1] - h1_features[:, 0]
    print(f"{title} - H1 lifetimes: {lifetimes}")

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
     
def convex_hull_metrics(points):
    """Calculate shape metrics for a point cloud using its convex hull."""
    if len(points) < 3:
        return {}  # Not enough points for meaningful hull
    
    hull = ConvexHull(points)
    
    # Basic metrics
    metrics = {
        'area': hull.volume,  # For 2D, volume=area
        'perimeter': 0.0,
        'num_vertices': len(hull.vertices)
    }
    
    # Calculate perimeter by summing edge lengths
    for simplex in hull.simplices:
        metrics['perimeter'] += np.linalg.norm(points[simplex[0]] - points[simplex[1]])
    
    # Derived metrics
    metrics['compactness'] = (4 * np.pi * metrics['area']) / (metrics['perimeter'] ** 2)
    return metrics

def distribution_distance(orig_points, aug_points):
    """Compare shape metrics between original and augmented distributions."""
    comparison = {}
    
    # Wasserstein distance between distributions
    comparison['wasserstein_x'] = wasserstein_distance(orig_points[:, 0], aug_points[:, 0])
    comparison['wasserstein_y'] = wasserstein_distance(orig_points[:, 1], aug_points[:, 1])
    
    # Sliced Wasserstein approximation
    try:
        comparison['sliced_wasserstein'] = sliced_wasserstein(orig_points, aug_points)
    except Exception as e:
        print(f"Sliced Wasserstein error: {str(e)}")
        comparison['sliced_wasserstein'] = np.nan

    # Convex hull metrics
    orig_metrics = convex_hull_metrics(orig_points)
    aug_metrics = convex_hull_metrics(aug_points)
    
    comparison['area_ratio'] = aug_metrics.get('area', 0) / orig_metrics.get('area', 1) if orig_metrics.get('area', 0) > 0 else 0
    comparison['perimeter_diff'] = aug_metrics.get('perimeter', 0) - orig_metrics.get('perimeter', 0)
    comparison['compactness_diff'] = aug_metrics.get('compactness', 0) - orig_metrics.get('compactness', 0)
    
    poly_orig = convex_hull_polygon(orig_points)
    poly_aug = convex_hull_polygon(aug_points)
    intersection_area = poly_orig.intersection(poly_aug).area
    left_non_overlap = poly_orig.area - intersection_area
    right_non_overlap = poly_aug.area - intersection_area
    comparison['left_non_overlap'] = left_non_overlap
    comparison['right_non_overlap'] = right_non_overlap
    comparison['total_non_overlap'] = left_non_overlap + right_non_overlap
    return comparison

def plot_metrics_comparison(metrics_dict, title):
    """Visualize metrics using bar chart."""
    plt.figure(figsize=(12, 6))
    labels = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    bars = plt.bar(range(len(values)), values)
    plt.title(title)
    plt.xticks(range(len(values)), labels, rotation=45)
    plt.grid(axis='y', alpha=0.5)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()

# New helper function: Compute convex hull as a Shapely Polygon
def convex_hull_polygon(points):
    hull = ConvexHull(points)
    return Polygon(points[hull.vertices])

def cache_results(result, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent = 4)

def load_cached_results(filename):
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def save_pickle(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None 

if __name__ == "__main__":
    #set pca components here
    pca_components = 2

    # Load or create cache
    augmented_cache_file = "augmented_cache.json"
    if os.path.exists(augmented_cache_file):
        with open(augmented_cache_file, 'r', encoding='utf-8') as f:
            aug_cache = json.load(f)
    else:
        aug_cache = {}

    # Load data - using only the first 20 examples
    dataset = load_dataset("glue", "sst2", split="train[:11855]")
    #dataset = dataset.shuffle(seed=42)         # Shuffle for randomness
    #dataset = dataset.select(range(500))         # Select 500 random examples
    raw_sentences = [item["sentence"] for item in dataset]

    # Process each raw sentence for augmentation (do not clean yet)
    for i, raw_sentence in enumerate(raw_sentences):
        if raw_sentence in aug_cache:
            continue
        
        print(f"Processing line {i}: {raw_sentence[:60]}...")
        # Augment using raw sentence directly
        gpt_text = gpt_augment_text(raw_sentence)
        bt_zh = back_translate(raw_sentence, intermediate_language="zh-Hans")
        bt_de = back_translate(raw_sentence, intermediate_language="de")
        aug_cache[raw_sentence] = {"gpt": gpt_text, "bt_zh": bt_zh, "bt_de": bt_de}
        if i % 50 == 0:
            with open(augmented_cache_file, 'w', encoding='utf-8') as f:
                json.dump(aug_cache, f, ensure_ascii=False, indent=4)
            print(f"Partial save at line {i}")
    with open(augmented_cache_file, 'w', encoding='utf-8') as f:
        json.dump(aug_cache, f, ensure_ascii=False, indent=4)
    print("All sentences augmented and saved to augmented_cache.json") 

    # --- Cleaning Stage: Clean Raw and Augmented Outputs Separately ---
    raw_list = []
    gpt_list = []
    bt_zh_list = []
    bt_de_list = []
    for raw_sentence in raw_sentences:
        if raw_sentence in aug_cache:
            raw_list.append(raw_sentence)  # Keep the raw sentence as is
            gpt_list.append(aug_cache[raw_sentence]["gpt"])
            bt_zh_list.append(aug_cache[raw_sentence]["bt_zh"])
            bt_de_list.append(aug_cache[raw_sentence]["bt_de"])
        else:
            print(f"Missing augmented output for: {raw_sentence}")

    # Now, clean the texts for further processing (e.g., for embeddings)
    cleaned_raw = [nltk_clean_text(s) for s in raw_list]
    cleaned_gpt = [nltk_clean_text(s) for s in gpt_list]
    cleaned_bt_zh = [nltk_clean_text(s) for s in bt_zh_list]
    cleaned_bt_de = [nltk_clean_text(s) for s in bt_de_list]

    # --- Word Embeddings and Further Analysis ---
    # Load Word2Vec Model
    print("Loading Word2Vec model (GoogleNews-vectors-negative300.bin)...")
    word2vec = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    # Load or create word vectors cache
    word_vectors_cache_file = "word_vectors_cache.pkl"
    cached_word_vectors = load_pickle(word_vectors_cache_file)
    if cached_word_vectors is not None:
        word_embedding_cache = cached_word_vectors
        print("Loaded cached word vectors.")
    else:
        print("No cached word vectors found, starting fresh.")

    emb_original, oov_info_orig, valid_orig = gather_word_vectors(cleaned_raw, word2vec)
    emb_gpt, oov_info_gpt, valid_gpt = gather_word_vectors(cleaned_gpt, word2vec)
    emb_bt_zh, oov_info_bt_zh, valid_bt_zh = gather_word_vectors(cleaned_bt_zh, word2vec)
    emb_bt_de, oov_info_bt_de, valid_bt_de = gather_word_vectors(cleaned_bt_de, word2vec)
    save_pickle(word_embedding_cache, word_vectors_cache_file)

    # Visualization: PCA and Convex Hull Plots
    n_orig = emb_original.shape[0]
    n_gpt = emb_gpt.shape[0]
    n_bt_de = emb_bt_de.shape[0]
    n_bt_zh = emb_bt_zh.shape[0]
    
    # One PCA for ALL data
    all_data = np.concatenate([emb_original, emb_gpt, emb_bt_de, emb_bt_zh], axis=0)
    pca = PCA(n_components=pca_components)
    all_data_2d = pca.fit_transform(all_data)
    
    # Slice out each group
    orig_2d = all_data_2d[:n_orig]
    gpt_2d = all_data_2d[n_orig:n_orig + n_gpt]
    bt_zh_2d = all_data_2d[n_orig + n_gpt:n_orig + n_gpt + n_bt_zh]
    bt_de_2d = all_data_2d[n_orig + n_gpt + n_bt_zh:n_orig + n_gpt + n_bt_zh + n_bt_de]

    metrics = {
        'Original': convex_hull_metrics(orig_2d),
        'GPT': convex_hull_metrics(gpt_2d),
        'BT-ZH': convex_hull_metrics(bt_zh_2d),
        'BT-DE': convex_hull_metrics(bt_de_2d)
    }

    # Plot area comparison
    area_data = {k: v['area'] for k, v in metrics.items()}
    plot_metrics_comparison(area_data, "Convex Hull Area Comparison")

    comparisons = {
        'GPT vs Original': distribution_distance(orig_2d, gpt_2d),
        'BT-ZH vs Original': distribution_distance(orig_2d, bt_zh_2d),
        'BT-DE vs Original': distribution_distance(orig_2d, bt_de_2d)
    }

    #print numerical results
    print("\n=== Shape Metrics ===")
    for name, data in metrics.items():
        print(f"\n{name}:")
        print(f"  Area: {data['area']:.2f}")
        print(f"  Perimeter: {data['perimeter']:.2f}")
        print(f"  Compactness: {data['compactness']:.3f}")
        print(f"  Vertices: {data['num_vertices']}")

    print("\n=== Distribution Distances ===")
    for name, data in comparisons.items():
        print(f"\n{name}:")
        print(f"  Wasserstein X: {data['wasserstein_x']:.4f}")
        print(f"  Wasserstein Y: {data['wasserstein_y']:.4f}")
        print(f"  Sliced Wasserstein: {data['sliced_wasserstein']:.4f}")
        print(f"  Advanced Multivariate Wasserstein: {data.get('multivariate_wasserstein', np.nan):.4f}")
        print(f"  Area Ratio: {data['area_ratio']:.2f}")
        print(f"  Perimeter Diff: {data['perimeter_diff']:.2f}")

    # Persistent Homology Analysis (always plotted)
    print("\n=== Persistent Homology ===")
    persistent_homology_analysis(orig_2d, "PH: Original Words", visualize=True)
    persistent_homology_analysis(gpt_2d, "PH: GPT Words", visualize=True)
    persistent_homology_analysis(bt_zh_2d, "PH: BT Chinese Words", visualize=True)
    persistent_homology_analysis(bt_de_2d, "PH: BT German Words", visualize=True)

    # If PCA dimension is 2, perform visualization and overlapping area calculations.
    if pca_components == 2:
    # === Overlapping Area Calculations ===
        # Compute convex hull polygons using our helper function
        poly_orig = convex_hull_polygon(orig_2d)
        poly_gpt = convex_hull_polygon(gpt_2d)
        poly_bt_zh = convex_hull_polygon(bt_zh_2d)
        poly_bt_de = convex_hull_polygon(bt_de_2d)

        # Original vs GPT
        intersection_orig_gpt = poly_orig.intersection(poly_gpt).area
        union_orig_gpt = poly_orig.union(poly_gpt).area
        non_overlap_orig_gpt = union_orig_gpt - intersection_orig_gpt

        # Original vs BT-ZH
        intersection_orig_bt_zh = poly_orig.intersection(poly_bt_zh).area
        union_orig_bt_zh = poly_orig.union(poly_bt_zh).area
        non_overlap_orig_bt_zh = union_orig_bt_zh - intersection_orig_bt_zh

        # Original vs BT-DE
        intersection_orig_bt_de = poly_orig.intersection(poly_bt_de).area
        union_orig_bt_de = poly_orig.union(poly_bt_de).area
        non_overlap_orig_bt_de = union_orig_bt_de - intersection_orig_bt_de

        print("\n=== Overlapping Areas between Convex Hulls ===")
        print(f"Original vs GPT:")
        print(f"  Intersection (Overlapping) Area: {intersection_orig_gpt:.4f}")
        print(f"  Union Area: {union_orig_gpt:.4f}")
        print(f"  Non-overlapping Area: {non_overlap_orig_gpt:.4f}")
        
        print(f"\nOriginal vs BT-ZH:")
        print(f"  Intersection (Overlapping) Area: {intersection_orig_bt_zh:.4f}")
        print(f"  Union Area: {union_orig_bt_zh:.4f}")
        print(f"  Non-overlapping Area: {non_overlap_orig_bt_zh:.4f}")
        
        print(f"\nOriginal vs BT-DE:")
        print(f"  Intersection (Overlapping) Area: {intersection_orig_bt_de:.4f}")
        print(f"  Union Area: {union_orig_bt_de:.4f}")
        print(f"  Non-overlapping Area: {non_overlap_orig_bt_de:.4f}")

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
    else: 
        print(f"\nVisualization and overlapping area calculations are skipped because PCA components = {pca_components} (only quantitative outputs computed).")

    results = {
    "pca_components": pca_components,
    "shape_metrics": metrics,
    "distribution_distances": comparisons,
    "persistent_homology": {
        "Original": {
            "H0_features": len(ripser(orig_2d)['dgms'][0]),
            "H1_features": len(ripser(orig_2d)['dgms'][1]),
            "H1_lifetimes": ripser(orig_2d)['dgms'][1].tolist()
        },
        "GPT": {
            "H0_features": len(ripser(gpt_2d)['dgms'][0]),
            "H1_features": len(ripser(gpt_2d)['dgms'][1]),
            "H1_lifetimes": ripser(gpt_2d)['dgms'][1].tolist()
        },
        "BT_Chinese": {
            "H0_features": len(ripser(bt_zh_2d)['dgms'][0]),
            "H1_features": len(ripser(bt_zh_2d)['dgms'][1]),
            "H1_lifetimes": ripser(bt_zh_2d)['dgms'][1].tolist()
        },
        "BT_German": {
            "H0_features": len(ripser(bt_de_2d)['dgms'][0]),
            "H1_features": len(ripser(bt_de_2d)['dgms'][1]),
            "H1_lifetimes": ripser(bt_de_2d)['dgms'][1].tolist()
        }
    },
    "overlapping_areas": {
        "GPT_vs_Original": {
            "intersection": intersection_orig_gpt,
            "union": union_orig_gpt,
            "non_overlapping": non_overlap_orig_gpt
        },
        "BT_Chinese_vs_Original": {
            "intersection": intersection_orig_bt_zh,
            "union": union_orig_bt_zh,
            "non_overlapping": non_overlap_orig_bt_zh
        },
        "BT_German_vs_Original": {
            "intersection": intersection_orig_bt_de,
            "union": union_orig_bt_de,
            "non_overlapping": non_overlap_orig_bt_de
        }
    }
}
    output_filename = "analysis_results.json"
    with open(output_filename, "w", encoding="utf-8") as outfile:
        json.dump(results, outfile, indent=4)
    print(f"Results have been written to {output_filename}")

    # Show all plots
    plt.show()
    print("Done. Close all windows to exit.")


