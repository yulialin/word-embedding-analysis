import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from sklearn.model_selection import KFold, train_test_split
import numpy as np
from collections import Counter
import pickle
from torch.utils.data import TensorDataset, DataLoader, Subset
import nltk
from nltk.tokenize import word_tokenize
import lightning as L
import textstat 
from gensim.models import KeyedVectors
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# load the aligned, cleaned and indexed data.
with open("aligned_data.pkl", "rb") as f:
    aligned_data = pickle.load(f)

# load previously stored word vector
with open("word_vectors_cache.pkl", "rb") as f:
    word2vec_dict = pickle.load(f)
print(f"Loaded {len(word2vec_dict)} word vectors from cache.")
# use gpu if avilable, else cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

aligned_cleaned_raw = aligned_data["aligned_cleaned_raw"]
aligned_cleaned_gpt = aligned_data["aligned_cleaned_gpt"]
aligned_cleaned_bt_zh = aligned_data["aligned_cleaned_bt_zh"]
aligned_cleaned_bt_de = aligned_data["aligned_cleaned_bt_de"]
aligned_labels = aligned_data["aligned_labels"]


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

# creates a dictionary mapping words to unique integer indexes based on their frequency
def build_vocab(texts, max_words=15000):
    """
    Build a vocabulary dictionary from texts.
    Reserve index 0 for padding and 1 for OOV.
    """
    counter = Counter()
    for text in texts:
        tokens = word_tokenize(text.lower())
        counter.update(tokens)
    most_common = counter.most_common(max_words)
    vocab = {word: i + 2 for i, (word, _) in enumerate(most_common)}
    vocab["<PAD>"] = 0 # pad frequency to a fixed length
    vocab["<OOV>"] = 1 # used for words that are OOV
    return vocab

# converts sentences into lists of integers based on the vocabulary mapping.
def texts_to_sequences(texts, vocab):
    """
    Convert list of texts to list of integer sequences using the vocabulary.
    """
    sequences = []
    for text in texts:
        tokens = word_tokenize(text.lower())
        seq = [vocab.get(token, vocab["<OOV>"]) for token in tokens]
        sequences.append(seq)
    return sequences

# pad setneces to fixed length for batches
def pad_sequences_custom(sequences, maxlen, padding="post", truncating="post", value=0):
    """
    Pad each sequence to the same length (maxlen).
    """
    padded = np.full((len(sequences), maxlen), value, dtype="int64")
    for i, seq in enumerate(sequences):
        if len(seq) > maxlen:
            seq = seq[:maxlen] if truncating == "post" else seq[-maxlen:]
        if padding == "post":
            padded[i, :len(seq)] = seq
        else:
            padded[i, -len(seq):] = seq
    return padded


# define the PyTorch LSTM Classifier using Lightning.
class LSTMClassifier(L.LightningModule):
    # constructor
    def __init__(self, hidden_dim, pad_idx, pretrained_embeddings):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=True, padding_idx = pad_idx)
        self.lstm = nn.LSTM(pretrained_embeddings.shape[1], hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_dim, 1)
    
    # pass the input text (as word-index sequences) through the embedding layer
    # feeds it to the lstm, applies dropout in the hidden state
    # then use the fc layer to predict final class scores 
    def forward(self, text): 
        # text shape: (batch, seq_len)
        embedded = self.embedding(text)                # (batch, seq_len, embedding_dim)
        _, (hidden, _) = self.lstm(embedded)   # hidden: (1, batch, hidden_dim)
       # hidden = hidden.squeeze(0)                       # (batch, hidden_dim)
        hidden = self.dropout(hidden.squeeze(0))
        return self.fc(hidden)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.0001)
    
    def training_step(self, batch, batch_idx):
        text, labels = batch
        logits = self(text).squeeze(1)
        probs = torch.sigmoid(logits)
        labels = labels.float()  # convert to float for BCELoss
        loss = nn.functional.binary_cross_entropy(probs, labels)
        
        # debug print for first batch
        if batch_idx == 0:
            print("\n=== [DEBUG] Training Step ===")
            print(f"Logits: {logits[:5].detach().cpu().numpy()}")
            print(f"Probs: {probs[:5].detach().cpu().numpy()}")
            print(f"Labels: {labels[:5].detach().cpu().numpy()}")
            print(f"Loss: {loss.item():.4f}")
            print("[DEBUG] Input batch (first 3 rows):")
            print(text[:3])
            print("[DEBUG] Unique token IDs in batch:", torch.unique(text))
            print(self.fc.bias)  

        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        text, labels = batch
        logits = self(text).squeeze(1)
        probs = torch.sigmoid(logits)
        labels = labels.float()
        loss = nn.functional.binary_cross_entropy(probs, labels)
        preds = (probs > 0.5).float()
        acc = (preds == labels).float().mean()

        if batch_idx == 0:
            print("\n=== [DEBUG] Validation Step ===")
            print(f"Logits: {logits[:5].detach().cpu().numpy()}")
            print(f"Probs: {probs[:5].detach().cpu().numpy()}")
            print(f"Labels: {labels[:5].detach().cpu().numpy()}")
            print(f"Preds: {preds[:5].detach().cpu().numpy()}")
            print(f"Loss: {loss.item():.4f}, Acc: {acc.item():.4f}")
            print("[DEBUG] FC bias (first):", self.fc.bias[:5].detach().cpu().numpy())

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return {"val_loss": loss, "val_acc": acc}
    
    def on_after_backward(self):
        if self.trainer.global_step % 1000 == 0: 
            print("\n=== [DEBUG] Gradient Flow Check ===")
            for name, param in self.named_parameters():
                if param.grad is not None:
                    grad_mean = param.grad.abs().mean().item()
                    print(f"{name}: grad mean = {grad_mean:.6f}")
                else: 
                    print(f"{name}: No gradient")

# build a combined vocabulary from all texts to capture augmentation-specific tokens.
all_texts = (aligned_cleaned_raw + aligned_cleaned_gpt +
             aligned_cleaned_bt_zh + aligned_cleaned_bt_de)

vocab = build_vocab(all_texts, max_words=15000)
vocab_counter = Counter()
for sentence in all_texts:
    tokens = word_tokenize(sentence.lower())
    vocab_counter.update(tokens)

unique_words = set(vocab_counter.keys())

words_with_vectors = [word for word in unique_words if word in word2vec_dict]
oov_words = [word for word in unique_words if word not in word2vec_dict]

print("Total unique words:", len(unique_words))
print("Words with Word2Vec embeddings:", len(words_with_vectors))
print("Out-of-vocabulary words:", len(oov_words))
print("OOV ratio: {:.2f}%".format(len(oov_words) / len(unique_words) * 100))

# define variants: for augmented variants, combine the original texts with the augmentation.
variants = {
    "Original": aligned_cleaned_raw,
    "Original+GPT": aligned_cleaned_raw + aligned_cleaned_gpt,
    "Original+BT-Chinese": aligned_cleaned_raw + aligned_cleaned_bt_zh,
    "Original+BT-German": aligned_cleaned_raw + aligned_cleaned_bt_de
}

# for non-orignal variants, double the amount of the labels: 
def get_variant_labels(variant_name, original_labels):
    if variant_name == "Original":
        return original_labels
    else: 
        return original_labels + original_labels

# parameters for Padding and Model
max_len = 30         # maximum length for each text sequence
batch_size = 16
embedding_dim = 300    
hidden_dim = 128
vocab_size = len(vocab)
pad_idx = vocab["<PAD>"]

# initialize embedding matrix with random values using normal distribution
embedding_matrix = np.random.normal(0, 1, (vocab_size, embedding_dim)).astype(np.float32)

# replace with pre-trained Word2Vec embeddings where available
for word, idx in vocab.items():
    # check if the word exists in Word2Vec vocabulary
    if word in word2vec_dict:
        embedding_matrix[idx] = word2vec_dict[word]

# convert the weight matrix into a torch tensor
pretrained_embedding_tensor = torch.tensor(embedding_matrix, dtype=torch.float32)
print(f"Pretrained embeddings shape: {pretrained_embedding_tensor.shape}")

# define test split ratio : setting 20% of data for testing 
test_split = 0.2

# set up 5-Fold cross-validation parameters 
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

results = {}

# compute and print basic statistics 
stats_original = compute_stats(aligned_cleaned_raw)
stats_gpt = compute_stats(aligned_cleaned_gpt)
stats_bt_zh = compute_stats(aligned_cleaned_bt_zh)
stats_bt_de = compute_stats(aligned_cleaned_bt_de)

print(f"[DEBUG] Original Stats: {stats_original}")
print(f"[DEBUG] GPT Stats: {stats_gpt}")
print(f"[DEBUG] BT-Chinese Stats: {stats_bt_zh}")
print(f"[DEBUG] BT-German Stats: {stats_bt_de}")

# print token count statistics
token_counts = [len(text.split()) for text in aligned_cleaned_raw]
print("[DEBUG] Original Token Count Statistics:")
print(f"  Min: {np.min(token_counts)}, Max: {np.max(token_counts)}, Mean: {np.mean(token_counts):.2f}")


label_counts = Counter(aligned_labels)
print(f"[DEBUG] Class distribution: {label_counts}")
minority_class = min(label_counts, key=label_counts.get)
print(f"[DEBUG] Minority class identified: {minority_class}")


# for each variants:
# - converts text to sequences and pads them
# - creates a TensorDataset and splits it into training and test sets
# - use 5-fold cv to train and validate on the training set
# - trains a final (model_final)  on the entire training set and evaluates it on the test set.
# - record results (cv + accuracy)
for variant_name, variant_texts in variants.items():
    print(f"\n====================")
    print(f"Processing variant: {variant_name}")
    print(f"====================\n")
    
    # get corresponding labels:
    variant_labels = get_variant_labels(variant_name, aligned_labels)
    
    # convert texts to sequences and pad them
    sequences = texts_to_sequences(variant_texts, vocab)
    padded_sequences = pad_sequences_custom(sequences, maxlen=max_len,
                                             padding="post", truncating="post",
                                             value=pad_idx)
    print(f"[DEBUG] Variant '{variant_name}': Number of texts = {len(padded_sequences)}, Number of labels = {len(variant_labels)}")
    print("[DEBUG] Sample input sequence before padding:", sequences[0])
    print("[DEBUG] Sample padded sequence:", padded_sequences[0]) 

    # convert to PyTorch tensors
    inputs_tensor = torch.LongTensor(padded_sequences)
    targets_tensor = torch.LongTensor(variant_labels)
    
    # create a TensorDataset
    dataset = TensorDataset(inputs_tensor, targets_tensor)
    
    # split the dataset into training and fixed test sets
    indices = np.arange(len(targets_tensor))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_split, random_state=42, stratify=targets_tensor.numpy()
    )
    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)
    
    # evaluate using K-Fold on the training set
    cv_fold_results = []
    train_indices_cv = np.arange(len(train_dataset))
    
    # convert Subset indices to list for splitting.
    for fold, (cv_train_idx, cv_val_idx) in enumerate(kf.split(train_indices_cv), 1):
        print(f"Starting CV fold {fold}/{n_splits} for variant '{variant_name}'...")
        # get CV train and validation subsets.
        cv_train_subset = Subset(train_dataset, cv_train_idx)
        cv_val_subset = Subset(train_dataset, cv_val_idx)
        
        train_loader = DataLoader(cv_train_subset, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(cv_val_subset, batch_size=batch_size, shuffle=False)
        
        # initialize a fresh model for this fold
        model = LSTMClassifier(hidden_dim=hidden_dim, pad_idx=pad_idx, pretrained_embeddings=pretrained_embedding_tensor)
        
        # create a Lightning Trainer
        trainer = L.Trainer(
            max_epochs=50,
            devices=1 if torch.cuda.is_available() else 0,
            log_every_n_steps=10,
            enable_checkpointing=False
            )
        
        # train and validate on the current fold
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        fold_metrics = trainer.validate(model, dataloaders=val_loader)

        # evaluate on the same loader to compute confusion matrix
        all_preds = []
        all_labels = []

        model = model.to(device)

        model.eval()
        for batch in val_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                logits = model(x).squeeze(1)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        print("Confusion Matrix:")
        print(confusion_matrix(all_labels, all_preds))
        print("Precision:", precision_score(all_labels, all_preds, zero_division=0))
        print("Recall:", recall_score(all_labels, all_preds, zero_division=0))
        print("F1 Score:", f1_score(all_labels, all_preds, zero_division=0))

        fold_val_acc = fold_metrics[0]["val_acc"]
        cv_fold_results.append(fold_val_acc)
        print(f"CV Fold {fold} validation accuracy for {variant_name}: {fold_val_acc*100:.2f}%\n")
    
    avg_cv_acc = np.mean(cv_fold_results)
    std_cv_acc = np.std(cv_fold_results)
    
    # evaluate the final model on the fixed test set
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # for test evaluation, retrain a fresh model on the entire training split
    model_final = LSTMClassifier(hidden_dim=hidden_dim, pad_idx=pad_idx, pretrained_embeddings=pretrained_embedding_tensor)
    
    trainer_final = L.Trainer(
        max_epochs=50,
        devices=1 if torch.cuda.is_available() else 0,
        log_every_n_steps=10,
        enable_checkpointing=False
    )
    trainer_final.fit(model_final, train_dataloaders=DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
                       val_dataloaders=DataLoader(train_dataset, batch_size=batch_size, shuffle=False))
    test_metrics = trainer_final.validate(model_final, dataloaders=test_loader)
    
    model_final = model_final.to(device)
    # evaluate on the same loader manually to compute confusion matrix
    all_preds = []
    all_labels = []

    model_final.eval()
    for batch in test_loader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            logits = model_final(x).squeeze(1)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("Precision:", precision_score(all_labels, all_preds, zero_division=0))
    print("Recall:", recall_score(all_labels, all_preds, zero_division=0))
    print("F1 Score:", f1_score(all_labels, all_preds, zero_division=0))

    test_acc = test_metrics[0]["val_acc"]
    
    results[variant_name] = {"CV_accuracy": (avg_cv_acc, std_cv_acc), "Test_accuracy": test_acc}
    print(f"Variant '{variant_name}': CV Accuracy = {avg_cv_acc*100:.2f}% ± {std_cv_acc*100:.2f}%")
    print(f"Variant '{variant_name}': Test Set Accuracy = {test_acc*100:.2f}%\n")


print("=== Final Comparison Across Variants ===")
for variant, metrics in results.items():
    cv_avg, cv_std = metrics["CV_accuracy"]
    test_acc = metrics["Test_accuracy"]
    print(f"{variant}: CV Accuracy = {cv_avg*100:.2f}% ± {cv_std*100:.2f}%, Test Accuracy = {test_acc*100:.2f}%")

print("\n=== DEBUG: Sample Cleaned Texts ===")
sample_indices = [0, 100, 500]  # choose random indices to see if label and sentences align across all diff variants 
for idx in sample_indices:
    if idx < len(aligned_cleaned_raw):
        print(f"Original[{idx}]: {aligned_cleaned_raw[idx]}")
    if idx < len(aligned_cleaned_gpt):
        print(f"GPT[{idx}]: {aligned_cleaned_gpt[idx]}")
    if idx < len(aligned_cleaned_bt_zh):
        print(f"BT-Chinese[{idx}]: {aligned_cleaned_bt_zh[idx]}")
    if idx < len(aligned_cleaned_bt_de):
        print(f"BT-German[{idx}]: {aligned_cleaned_bt_de[idx]}")
    print("-" * 80)
