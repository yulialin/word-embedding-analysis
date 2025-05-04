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
from openai import OpenAI
from scipy.spatial import ConvexHull, Delaunay
from scipy.stats import wasserstein_distance, entropy 
import math
from shapely.geometry import Polygon
import pickle
import ot 
import string
import textstat 

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))

# API keys and endpoints
os.environ['OPENAI_API_KEY'] = "REPLACE WITH YOUR OWN KEY"
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

# Azure Translator API settings
AZURE_SUBSCRIPTION_KEY = "REPLACE WITH YOUR OWN KEY"
AZURE_ENDPOINT = "https://api.cognitive.microsofttranslator.com"
AZURE_LOCATION = "eastus"

def translate_text(text, from_language="en", to_languages=["fr"]):
    # translates text using the Azure Translator API.
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
    # performs back-translation: English -> Intermediate Language -> English.
    
    # translate to the intermediate language
    intermediate_result = translate_text(text, to_languages=[intermediate_language])
    translated_text = intermediate_result[0]['translations'][0]['text']
    # translate back to English
    back_result = translate_text(translated_text, from_language=intermediate_language, to_languages=["en"])
    return back_result[0]['translations'][0]['text']


# GPT augmentation function
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


# data cleaning using nltk
def nltk_clean_text(text):
    if text is None:
        return ""
    # tokenize text into sentences
    sentences = sent_tokenize(text)
    cleaned_sentences = []
    
    for sentence in sentences:
        #tokenize sentence into words
        words = word_tokenize(sentence)
        cleaned_words = []
        for word in words:
            #remove all punctuation characters using regex
            cleaned_word = re.sub(r'[^\w\s]', ' ', word)
            cleaned_word = cleaned_word.lower().strip()
            # remove extra spaces if any appear due to the replacement
            cleaned_word = re.sub(r'\s+', ' ', cleaned_word)
            # filter out empty tokens and stopwords
            if cleaned_word and cleaned_word not in stop_words:
                cleaned_words.append(cleaned_word)
        # only add non-empty sentences
        if cleaned_words:
            cleaned_sentences.append(" ".join(cleaned_words))
    
    # return the joined cleaned sentences (or an empty string if none)
    return " ".join(cleaned_sentences) if cleaned_sentences else ""

# get word embeddings for Individual Words using Word2Vec
# adpating the og paper style, gathering all sentence-level arrays into one big array
# also cache for out-of-vobulary words that doesn't have a word vecotor 
word_embedding_cache = {}
oov_words_set = set()

# global count to track sentences dropped due to high OOV ratio
oov_dropped = {
    'total_dropped': 0,
    'details':[] 
}

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
            # print(f"Word '{word}' not found in the Word2Vec vocabulary.")
    return np.array(word_vectors), oov_count, len(words)

# changed: gather aligned word vectors across all groups
def gather_aligned_word_vectors(cleaned_raw, cleaned_gpt, cleaned_bt_zh, cleaned_bt_de, model, threshold=0.3):
    aligned_raw = []
    aligned_gpt = []
    aligned_bt_zh = []
    aligned_bt_de = []
    aligned_indices = []
    oov_info_aligned = []

    for i in range(len(cleaned_raw)):
        vec_orig, oov_count_orig, total_orig = get_word_vectors(cleaned_raw[i], model)
        vec_gpt, oov_count_gpt, total_gpt = get_word_vectors(cleaned_gpt[i], model)
        vec_bt_zh, oov_count_bt_zh, total_bt_zh = get_word_vectors(cleaned_bt_zh[i], model)
        vec_bt_de, oov_count_bt_de, total_bt_de = get_word_vectors(cleaned_bt_de[i], model)
        
        ratio_orig = oov_count_orig / total_orig if total_orig > 0 else 0
        ratio_gpt = oov_count_gpt / total_gpt if total_gpt > 0 else 0
        ratio_bt_zh = oov_count_bt_zh / total_bt_zh if total_bt_zh > 0 else 0
        ratio_bt_de = oov_count_bt_de / total_bt_de if total_bt_de > 0 else 0
        
        if (vec_orig.size > 0 and vec_gpt.size > 0 and vec_bt_zh.size > 0 and vec_bt_de.size > 0 and 
            ratio_orig <= threshold and ratio_gpt <= threshold and ratio_bt_zh <= threshold and ratio_bt_de <= threshold):
            aligned_raw.append(vec_orig)
            aligned_gpt.append(vec_gpt)
            aligned_bt_zh.append(vec_bt_zh)
            aligned_bt_de.append(vec_bt_de)
            aligned_indices.append(i)
            oov_info_aligned.append({
                "index": i,
                "orig": (oov_count_orig, total_orig),
                "gpt": (oov_count_gpt, total_gpt),
                "bt_zh": (oov_count_bt_zh, total_bt_zh),
                "bt_de": (oov_count_bt_de, total_bt_de)
            })
        else:
            # log which sentence is dropped
            oov_dropped['total_dropped'] += 1
            oov_dropped['details'].append({
                "index": i,
                "orig": (oov_count_orig, total_orig),
                "gpt": (oov_count_gpt, total_gpt),
                "bt_zh": (oov_count_bt_zh, total_bt_zh),
                "bt_de": (oov_count_bt_de, total_bt_de)
            })
    if len(aligned_raw) > 0:
        emb_orig = np.vstack(aligned_raw)
        emb_gpt = np.vstack(aligned_gpt)
        emb_bt_zh = np.vstack(aligned_bt_zh)
        emb_bt_de = np.vstack(aligned_bt_de)
    else:
        emb_orig = np.array([]).reshape(0,300)
        emb_gpt = np.array([]).reshape(0,300)
        emb_bt_zh = np.array([]).reshape(0,300)
        emb_bt_de = np.array([]).reshape(0,300)
    return emb_orig, emb_gpt, emb_bt_zh, emb_bt_de, aligned_indices, oov_info_aligned

# 2-Wasserstein distance 
def compute_multivariate_wasserstein(X, Y, sample_size=10000):
    # Downsample if necessary
    if X.shape[0] > sample_size:
        X = X[:sample_size]        
    if Y.shape[0] > sample_size:
        Y = Y[:sample_size]
    
    M = ot.dist(X, Y) # reutrns suqared Euclidean of x and y 
    a = np.ones(X.shape[0]) / X.shape[0]
    b = np.ones(Y.shape[0]) / Y.shape[0]
    distance2 = ot.emd2(a, b, M)
    return np.sqrt(distance2)

# persistent Homology and visualization
def persistent_homology_analysis(vectors, title, visualize=False, sample_size=20000, sample_idx=None):
    # subsetting
    if sample_idx is not None:
        vectors = vectors[sample_idx]
    else:
        n = len(vectors)
        if n > sample_size:
            idx = np.random.choice(n, size=sample_size, replace=False)
            vectors = vectors[idx]

    if len(vectors) > sample_size:
        idx = np.random.choice(n, size=sample_size, replace=False)
        vectors = vectors[idx]
    
    # compute persistent homology up to H1
    results = ripser(vectors, maxdim=1)
    diagrams = results.get('dgms', [])

    if visualize:
        plt.figure()
        persim.plot_diagrams(diagrams, title=title)
    
    h0_features = diagrams[0] if len(diagrams) > 0 else np.array([])
    h1_features = diagrams[1] if len(diagrams) > 1 else np.array([])
    print(f"{title} - H0 features count: {len(h0_features)}")
    print(f"{title} - H1 features count: {len(h1_features)}")
    
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
     
def convex_hull_metrics(points):
    """Calculate shape metrics for a point cloud using its convex hull."""
    if len(points) < 3:
        return {}  # Not enough points for meaningful hull
    
    hull = ConvexHull(points)

    metrics = {
        'area': hull.volume,  # For 2D, volume=area
        'perimeter': 0.0,
        'num_vertices': len(hull.vertices)
    }
    
    # calculate perimeter by summing edge lengths
    for simplex in hull.simplices:
        metrics['perimeter'] += np.linalg.norm(points[simplex[0]] - points[simplex[1]])
    
    metrics['compactness'] = (4 * np.pi * metrics['area']) / (metrics['perimeter'] ** 2)
    return metrics

# generalized convex hull metrics for higher PCA dimensions
def convex_hull_metrics_general(points, pca_components=3):
    """Calculate convex hull metrics for 3D data: volume, surface area, and number of vertices."""
    if len(points) < pca_components:
        return {}
    hull = ConvexHull(points, qhull_options="QJ")
    metrics = {
        "volume": hull.volume,       
        "surface_area": hull.area,      
        "num_vertices": len(hull.vertices)
    }
    return metrics
    
def delaunay_avg_edge_length(points):
    tri = Delaunay(points)
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            edge = tuple(sorted((simplex[i], simplex[(i+1)%3])))
            edges.add(edge)
        
    edge_lengths = [np.lialg.norm(points[i]-points[j]) for i, j in edges]
    avg_length = np.mean(edge_lengths)
    return avg_length 
        
def distribution_distance(orig_points, aug_points):
    """Compare shape metrics between original and augmented distributions."""
    comparison = {}
    d = orig_points.shape[1]
    
    # wasserstein Distances
    comparison['wasserstein_x'] = wasserstein_distance(orig_points[:, 0], aug_points[:, 0])
    comparison['wasserstein_y'] = wasserstein_distance(orig_points[:, 1], aug_points[:, 1])
    if d == 3:
        comparison['wasserstein_z'] = wasserstein_distance(orig_points[:, 2], aug_points[:, 2])
    # multivariate Wasserstein
    comparison['multivariate_wasserstein'] = compute_multivariate_wasserstein(orig_points, aug_points)
    # convex Hull metrics 
    if d == 2:
        # 2D-specific metrics 
        orig_metrics = convex_hull_metrics(orig_points)
        aug_metrics = convex_hull_metrics(aug_points)
        
        comparison['area_ratio'] = aug_metrics['area'] / orig_metrics['area'] if orig_metrics['area'] > 0 else 0
        comparison['perimeter_diff'] = aug_metrics['perimeter'] - orig_metrics['perimeter']
        
        poly_orig = convex_hull_polygon(orig_points)
        poly_aug = convex_hull_polygon(aug_points)
        
        intersection = poly_orig.intersection(poly_aug).area
        union = poly_orig.union(poly_aug).area
        non_overlap = union - intersection
        
        comparison['intersection_area'] = intersection
        comparison['union_area'] = union
        comparison['non_overlapping_area'] = non_overlap    
    elif d == 3:
        # 3D-specific metrics
        orig_metrics = convex_hull_metrics_general(orig_points, 3)
        aug_metrics = convex_hull_metrics_general(aug_points, 3)
        
        comparison['volume_ratio'] = aug_metrics['volume'] / orig_metrics['volume'] if orig_metrics['volume'] > 0 else 0
        comparison['surface_area_diff'] = aug_metrics['surface_area'] - orig_metrics['surface_area']
    
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
    
    # add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()

# compute convex hull as a Shapely Polygon
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

# compute basic statistics and readability for a list of sentences.
def compute_stats(sentences):
    # filter out empty sentences
    valid_sentences = [s for s in sentences if s.strip() != '']    
    # compute sentence lengths (number of words)
    lengths = [len(s.split()) for s in valid_sentences]  
    # compute readability scores using Flesch-Kincaid grade level
    readability_scores = [textstat.flesch_kincaid_grade(s) for s in valid_sentences]
    
    stats_summary = {} 
    if lengths:
        stats_summary["average_length"] = np.mean(lengths)
        stats_summary["median_length"] = np.median(lengths)
        stats_summary["std_dev_length"] = np.std(lengths)
    else:
        stats_summary["average_length"] = stats_summary["median_length"] = stats_summary["std_dev_length"] = 0
    
    if readability_scores:
        stats_summary["average_readability"] = np.mean(readability_scores)
        stats_summary["median_readability"] = np.median(readability_scores)
        stats_summary["std_dev_readability"] = np.std(readability_scores)
    else:
        stats_summary["average_readability"] = stats_summary["median_readability"] = stats_summary["std_dev_readability"] = None 
    return stats_summary

# get border examples from PCA data (sentences farthest from centroid)
def get_border_examples(pca_data, sentences, num_examples=10):
    centroid = np.mean(pca_data, axis=0)
    distances = np.linalg.norm(pca_data - centroid, axis=1)
    sorted_indices = np.argsort(-distances)  # farthest first
    examples = [sentences[i] for i in sorted_indices[:num_examples]]
    return examples

# calculate KL Divergence between two distributions
def kl_divergence(dist_p, dist_q, epsilon=1e-10):
    p = np.array(dist_p) + epsilon
    q = np.array(dist_q) + epsilon
    p /= p.sum()
    q /= q.sum()
    return entropy(p, q)

# to print array info (data type, shape, etc.)
def print_array_info(arr, name="Array"):
    print(f"{name} - shape: {arr.shape}, dtype: {arr.dtype}, min: {arr.min()}, max: {arr.max()}")

# this could be edited to be more robust
def get_vocabulary(sentences):
    vocab = set()
    for sentence in sentences:
        words = sentence.split()
        vocab.update(words)
    return vocab

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
    raw_sentences = [item["sentence"] for item in dataset]
    labels = [item["label"] for item in dataset]

    # Process each raw sentence for augmentation (do not clean yet)
    for i, raw_sentence in enumerate(raw_sentences):
        if raw_sentence in aug_cache:
            continue
        
        # print(f"Processing line {i}: {raw_sentence[:60]}...")
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

    # clean the texts for further processing 
    cleaned_raw = [nltk_clean_text(s) for s in raw_list]
    cleaned_gpt = [nltk_clean_text(s) for s in gpt_list]
    cleaned_bt_zh = [nltk_clean_text(s) for s in bt_zh_list]
    cleaned_bt_de = [nltk_clean_text(s) for s in bt_de_list]

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

    # gather aligned word vectors across all groups 
    emb_original, emb_gpt, emb_bt_zh, emb_bt_de, aligned_indices, oov_info_aligned = gather_aligned_word_vectors(
        cleaned_raw, cleaned_gpt, cleaned_bt_zh, cleaned_bt_de, word2vec
    )
    save_pickle(word_embedding_cache, word_vectors_cache_file)

    # Define aligned_labels by filtering the original labels using aligned_indices
    aligned_labels = [labels[i] for i in aligned_indices]
    # print("Aligned Labels:", aligned_labels) 
    # this will be binary labels, 0 or 1 for positive or negative

    total_sentences = len(raw_list)
    dropped = oov_dropped['total_dropped']
    percentage = (dropped / total_sentences) * 100 if total_sentences > 0 else 0
    print(f"Total sentences dropped due to OOV (across groups): {dropped} out of {total_sentences} ({percentage:.2f}%)")

    # print basic numpy array info for debugging
    print_array_info(emb_original, "Original Embeddings")
    print_array_info(emb_gpt, "GPT Embeddings")
    print_array_info(emb_bt_zh, "BT Chinese Embeddings")
    print_array_info(emb_bt_de, "BT German Embeddings")

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
    bt_de_2d = all_data_2d[n_orig + n_gpt:n_orig + n_gpt + n_bt_de]
    bt_zh_2d = all_data_2d[n_orig + n_gpt + n_bt_de:n_orig + n_gpt + n_bt_de + n_bt_zh]

    if pca_components == 2:
        metrics = {
            'Original': convex_hull_metrics(orig_2d),
            'GPT': convex_hull_metrics(gpt_2d),
            'BT-ZH': convex_hull_metrics(bt_zh_2d),
            'BT-DE': convex_hull_metrics(bt_de_2d)
        }
        # Plot area comparison
        area_data = {k: v['area'] for k, v in metrics.items()}
        plot_metrics_comparison(area_data, "Convex Hull Area Comparison")
    elif pca_components == 3:
        metrics = {
            'Original': convex_hull_metrics_general(orig_2d, pca_components),
            'GPT': convex_hull_metrics_general(gpt_2d, pca_components),
            'BT-ZH': convex_hull_metrics_general(bt_zh_2d, pca_components),
            'BT-DE': convex_hull_metrics_general(bt_de_2d, pca_components)
        }
        volume_data = {k: v['volume'] for k, v in metrics.items()}
        plot_metrics_comparison(volume_data, "Convex Hull Volume Comparison")

    # print numerical results
    print("\n=== Shape Metrics ===")
    for name, data in metrics.items():
        print(f"\n{name}:")
        if pca_components == 2:
            print(f"  Area: {data.get('area', 0):.2f}")
            print(f"  Perimeter: {data.get('perimeter', 0):.2f}")
            print(f"  Compactness: {data.get('compactness', 0):.3f}")
            print(f"  Vertices: {data.get('num_vertices', 0)}")
        elif pca_components == 3:
            print(f"  Volume: {data.get('volume', 0):.2f}")
            print(f"  Surface Area: {data.get('surface_area', 0):.2f}")
            print(f"  Vertices: {data.get('num_vertices', 0)}")
        
    comparisons = {
        'GPT vs Original': distribution_distance(orig_2d, gpt_2d),
        'BT-ZH vs Original': distribution_distance(orig_2d, bt_zh_2d),
        'BT-DE vs Original': distribution_distance(orig_2d, bt_de_2d)
    }

    # print distribution distances 
    print("\n=== Distribution Distances ===")
    for name, data in comparisons.items():
        print(f"\n{name}:")
        if pca_components == 2:
            print(f"  Wasserstein X: {data.get('wasserstein_x', 0):.4f}")
            print(f"  Wasserstein Y: {data.get('wasserstein_y', 0):.4f}")
            print(f"  Sliced Wasserstein (H1): {data.get('sliced_wasserstein_h1', 0):.4f}")
            print(f"  Multivariate Wasserstein: {data.get('multivariate_wasserstein', 0):.4f}")
            print(f"  Area Ratio: {data.get('area_ratio', 0):.2f}")
            print(f"  Perimeter Diff: {data.get('perimeter_diff', 0):.2f}")
        elif pca_components == 3:
            print(f"  Wasserstein X: {data.get('wasserstein_x', 0):.4f}")
            print(f"  Wasserstein Y: {data.get('wasserstein_y', 0):.4f}")
            print(f"  Wasserstein Z: {data.get('wasserstein_z', 0):.4f}")
            print(f"  Multivariate Wasserstein: {data.get('multivariate_wasserstein', 0):.4f}")
            print(f"  Volume Ratio: {data.get('volume_ratio', 0):.2f}")
            print(f"  Surface Area Diff: {data.get('surface_area_diff', 0):.2f}")

    # Create aligned cleaned lists using the aligned indices from the OOV filtering
    aligned_cleaned_raw = [cleaned_raw[i] for i in aligned_indices]
    aligned_cleaned_gpt = [cleaned_gpt[i] for i in aligned_indices]
    aligned_cleaned_bt_zh = [cleaned_bt_zh[i] for i in aligned_indices]
    aligned_cleaned_bt_de = [cleaned_bt_de[i] for i in aligned_indices]

    vocab_orig = get_vocabulary(aligned_cleaned_raw)
    vocab_gpt = get_vocabulary(aligned_cleaned_gpt)
    vocab_bt_zh = get_vocabulary(aligned_cleaned_bt_zh)
    vocab_bt_de = get_vocabulary(aligned_cleaned_bt_de)

    print("\n=== Vocabulary Counts ===")
    print("Original vocabulary count:", len(vocab_orig))
    print("GPT vocabulary count:", len(vocab_gpt))
    print("BT-Chinese vocabulary count:", len(vocab_bt_zh))
    print("BT-German vocabulary count:", len(vocab_bt_de))

    new_words_gpt = vocab_gpt - vocab_orig
    new_words_bt_zh = vocab_bt_zh - vocab_orig
    new_words_bt_de = vocab_bt_de - vocab_orig

    print("New words in GPT augmentation:", len(new_words_gpt))
    print("New words in BT-Chinese augmentation:", len(new_words_bt_zh))
    print("New words in BT-German augmentation:", len(new_words_bt_de))

    # compute dataset statistics (sentence length and readability)
    stats_original = compute_stats(aligned_cleaned_raw)
    stats_gpt = compute_stats(aligned_cleaned_gpt)
    stats_bt_zh = compute_stats(aligned_cleaned_bt_zh)
    stats_bt_de = compute_stats(aligned_cleaned_bt_de)

    print("\n=== Dataset Statistics ===")
    print("Original Stats:", stats_original)
    print("GPT Stats:", stats_gpt)
    print("BT-Chinese Stats:", stats_bt_zh)
    print("BT-German Stats:", stats_bt_de)

    # calculate KL Divergence for sentence length distributions 
    lengths_orig = [len(s.split()) for s in aligned_cleaned_raw]
    lengths_gpt = [len(s.split()) for s in aligned_cleaned_gpt]
    lengths_bt_zh = [len(s.split()) for s in aligned_cleaned_bt_zh]
    lengths_bt_de = [len(s.split()) for s in aligned_cleaned_bt_de]

    hist_orig, _ = np.histogram(lengths_orig, bins=10)
    hist_gpt, _ = np.histogram(lengths_gpt, bins=10)
    hist_bt_zh, _ = np.histogram(lengths_bt_zh, bins=10)
    hist_bt_de, _ = np.histogram(lengths_bt_de, bins=10)

    kl_val_original_gpt = kl_divergence(hist_orig, hist_gpt)
    kl_val_original_bt_zh = kl_divergence(hist_orig, hist_bt_zh)
    kl_val_original_bt_de = kl_divergence(hist_orig, hist_bt_de)

    print(f"\nKL Divergence (Original vs GPT sentence lengths): {kl_val_original_gpt:.4f}")
    print(f"KL Divergence (Original vs BT-Chinese sentence lengths): {kl_val_original_bt_zh:.4f}")
    print(f"KL Divergence (Original vs BT-German sentence lengths): {kl_val_original_bt_de:.4f}")

    sample_size = 20000

    # figure out how many points each variant has
    n_orig   = orig_2d.shape[0]
    n_gpt    = gpt_2d.shape[0]
    n_btzh  = bt_zh_2d.shape[0]
    n_btde  = bt_de_2d.shape[0]
    print("total og pts: ", n_orig)
    print("total gpt pts: ", n_gpt)
    print("total bt zh pts: ", n_btzh)  
    print("total bt de pts: ", n_btde)  
    # pick the smallest one as your base
    base_n = min(n_orig, n_gpt, n_btzh, n_btde)

    # decide how many to sample
    smpl_n   = min(sample_size, base_n)

    # draw one shared random sample of indices from [0..base_n)
    rng      = np.random.default_rng(42)
    sample_idx = rng.choice(base_n, size=smpl_n, replace=False)

    # subset each variant using the same indices
    orig_sub = orig_2d[sample_idx]
    gpt_sub  = gpt_2d[sample_idx]
    btzh_sub = bt_zh_2d[sample_idx]
    btde_sub = bt_de_2d[sample_idx]

    # Persistent Homology Analysis (always plotted)
    print("\n=== Persistent Homology Analysis ===")
    diagrams_orig = persistent_homology_analysis(orig_sub, "PH: Original Words", visualize=False)
    diagrams_gpt = persistent_homology_analysis(gpt_sub, "PH: GPT Words", visualize=False)
    diagrams_bt_zh = persistent_homology_analysis(btzh_sub, "PH: BT Chinese Words", visualize=False)
    diagrams_bt_de = persistent_homology_analysis(btde_sub, "PH: BT German Words", visualize=False)
   
    print("H1 diagram lengths:")
    print("  Original:", len(diagrams_orig[1]))
    print("  GPT:", len(diagrams_gpt[1]))
    print("  BT-ZH:", len(diagrams_bt_zh[1]))
    print("  BT-DE:", len(diagrams_bt_de[1]))

    # Compute sliced Wasserstein distances between H1 diagrams
    sw_gpt = sliced_wasserstein(diagrams_orig[1], diagrams_gpt[1])
    sw_bt_zh = sliced_wasserstein(diagrams_orig[1], diagrams_bt_zh[1])
    sw_bt_de = sliced_wasserstein(diagrams_orig[1], diagrams_bt_de[1])

    comparisons = {
        'GPT vs Original': distribution_distance(orig_2d, gpt_2d),
        'BT-ZH vs Original': distribution_distance(orig_2d, bt_zh_2d),
        'BT-DE vs Original': distribution_distance(orig_2d, bt_de_2d)
    }

    # Update comparisons with sliced Wasserstein from persistence diagrams
    comparisons['GPT vs Original']['sliced_wasserstein_h1'] = sw_gpt
    comparisons['BT-ZH vs Original']['sliced_wasserstein_h1'] = sw_bt_zh
    comparisons['BT-DE vs Original']['sliced_wasserstein_h1'] = sw_bt_de

    print("Orig H1 Diagram:", diagrams_orig[1])
    print("GPT H1 Diagram:", diagrams_gpt[1])
    print("BT-ZH H1 Diagram:", diagrams_bt_zh[1])
    print("BT-DE H1 Diagram:", diagrams_bt_de[1])
    # perform visualization and overlapping area calculations
    if pca_components == 2:
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
        
        if poly_gpt.within(poly_orig):
            print("The GPT convex hull is completely within the original convex hull.")
        else:
            print("The GPT convex hull is not completely within the original convex hull.")

        print(f"\nOriginal vs BT-ZH:")
        print(f"  Intersection (Overlapping) Area: {intersection_orig_bt_zh:.4f}")
        print(f"  Union Area: {union_orig_bt_zh:.4f}")
        print(f"  Non-overlapping Area: {non_overlap_orig_bt_zh:.4f}")
        
        print(f"\nOriginal vs BT-DE:")
        print(f"  Intersection (Overlapping) Area: {intersection_orig_bt_de:.4f}")
        print(f"  Union Area: {union_orig_bt_de:.4f}")
        print(f"  Non-overlapping Area: {non_overlap_orig_bt_de:.4f}")

        # Plot 1: Original vs. Back-Translated (Chinese)
        #plt.figure(figsize=(10, 8))
        #plt.scatter(orig_2d[:, 0], orig_2d[:, 1], c='blue', marker='o', label='Original')
        #plt.scatter(bt_zh_2d[:, 0], bt_zh_2d[:, 1], c='orange', marker='s', label='Back-Translated (Chinese)')
        #plot_delaunay(orig_2d, color='blue')
        #plot_delaunay(bt_zh_2d, color='orange')
        #plot_convex_hull(orig_2d, 'blue')
        #plot_convex_hull(bt_zh_2d, 'orange')
        #plt.title("Word Embeddings (Single PCA): Original vs Back-Translated (Chinese)")
        #plt.xlabel("PCA Component 1")
        #plt.ylabel("PCA Component 2")
        #plt.legend()

        # Plot 2: Original vs. Back-Translated (German)
        #plt.figure(figsize=(10, 8))
        #plt.scatter(orig_2d[:, 0], orig_2d[:, 1], c='blue', marker='o', label='Original')
        #plt.scatter(bt_de_2d[:, 0], bt_de_2d[:, 1], c='red', marker='^', label='Back-Translated (German)')
        #plot_delaunay(orig_2d, color='blue')
        #plot_delaunay(bt_de_2d, color='red')
        #plot_convex_hull(orig_2d, 'blue')
        #plot_convex_hull(bt_de_2d, 'red')
        #plt.title("Word Embeddings (Single PCA): Original vs Back-Translated (German)")
        #plt.xlabel("PCA Component 1")
        #plt.ylabel("PCA Component 2")
        #plt.legend()

        # Plot 3: Original vs. GPT-Augmented
        #plt.figure(figsize=(10, 8))
        #plt.scatter(orig_2d[:, 0], orig_2d[:, 1], c='blue', marker='o', label='Original')
        #plt.scatter(gpt_2d[:, 0], gpt_2d[:, 1], c='green', marker='D', label='GPT-Augmented')
        #plot_delaunay(orig_2d, color='blue')
        #plot_delaunay(gpt_2d, color='green')
        #plot_convex_hull(orig_2d, 'blue')
        #plot_convex_hull(gpt_2d, 'green')
        #plt.title("Word Embeddings (Single PCA): Original vs GPT")
        #plt.xlabel("PCA Component 1")
        #plt.ylabel("PCA Component 2")
        #plt.legend()
    else: 
        print(f"\nVisualization and overlapping area calculations are skipped because PCA components = {pca_components} (only quantitative outputs computed).")
    
    ph_intervals = {
        "Original": {
            "H0": diagrams_orig[0].tolist(),
            "H1": diagrams_orig[1].tolist()
        },
        "GPT": {
            "H0": diagrams_gpt[0].tolist(),
            "H1": diagrams_gpt[1].tolist()
        },
        "BT_Chinese": {
            "H0": diagrams_bt_zh[0].tolist(),
            "H1": diagrams_bt_zh[1].tolist()
        },
        "BT_German": {
            "H0": diagrams_bt_de[0].tolist(),
            "H1": diagrams_bt_de[1].tolist()
        }
    }

    with open("ph_lifetimes.json", "w", encoding="utf-8") as f:
        json.dump(ph_intervals, f, indent=4)
    print("Persistent-homology lifetimes saved to ph_lifetimes.json")

    results = {
        "pca_components": pca_components,
        "shape_metrics": metrics,
        "distribution_distances": comparisons,
        "persistent_homology": {
            "Original": {
                "H0_features": len(diagrams_orig[0]),
                "H1_features": len(diagrams_orig[1]),
            },
            "GPT": {
                "H0_features": len(diagrams_gpt[0]),
                "H1_features": len(diagrams_gpt[1]),
            },
            "BT_Chinese": {
                "H0_features": len(diagrams_bt_zh[0]),
                "H1_features": len(diagrams_bt_zh[1]),
              },
            "BT_German": {
                "H0_features": len(diagrams_bt_de[0]),
                "H1_features": len(diagrams_bt_de[1]),
             }
        },
       "overlapping_areas": {
            "GPT_vs_Original": {
                "intersection": intersection_orig_gpt if pca_components == 2 else None,
                "union": union_orig_gpt if pca_components == 2 else None,
                "non_overlapping": non_overlap_orig_gpt if pca_components == 2 else None
            },
            "BT_Chinese_vs_Original": {
                "intersection": intersection_orig_bt_zh if pca_components == 2 else None,
                "union": union_orig_bt_zh if pca_components == 2 else None,
                "non_overlapping": non_overlap_orig_bt_zh if pca_components == 2 else None
            },
            "BT_German_vs_Original": {
                "intersection": intersection_orig_bt_de if pca_components == 2 else None,
                "union": union_orig_bt_de if pca_components == 2 else None,
                "non_overlapping": non_overlap_orig_bt_de if pca_components == 2 else None
            }
        },
        # dataset statistics and OOV dropped info
        "dataset_statistics": {
            "Original": stats_original,
            "GPT": stats_gpt,
            "BT_Chinese": stats_bt_zh,
            "BT_German": stats_bt_de
        },
        "oov_dropped": oov_dropped  # Information about dropped sentences due to high OOV ratio
    }
    output_filename = "analysis_results.json"
    with open(output_filename, "w", encoding="utf-8") as outfile:
        json.dump(results, outfile, indent=4)
    print(f"Results have been written to {output_filename}")

    aligned_data = {
    "aligned_cleaned_raw": aligned_cleaned_raw,
    "aligned_cleaned_gpt": aligned_cleaned_gpt,
    "aligned_cleaned_bt_zh": aligned_cleaned_bt_zh,
    "aligned_cleaned_bt_de": aligned_cleaned_bt_de,
    "aligned_labels": aligned_labels
    }
    with open("aligned_data.pkl", "wb") as f:
        pickle.dump(aligned_data, f)

    #plt.show()
    print("Done. Close all windows to exit.")


