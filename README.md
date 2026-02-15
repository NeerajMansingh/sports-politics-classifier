# Sports vs Politics Text Classifier

A comparative study of machine learning approaches for binary text classification, distinguishing between sports and politics-related documents from the 20 Newsgroups dataset.

## Project Overview

This project implements three classical machine learning algorithms from scratch to classify newsgroup posts as either sports or politics-related. We evaluate how different feature extraction methods (Bag of Words, Bigrams, TF-IDF) impact classification performance across Naive Bayes, Logistic Regression, and K-Nearest Neighbors classifiers.

**Best Performance:** 96.97% accuracy using Naive Bayes with Bag of Words features.

## Motivation

Text classification is fundamental to many NLP applications. This project explores:
- Whether simple feature representations can outperform complex ones
- How different algorithms respond to various feature engineering choices
- The practical tradeoffs between model complexity and performance

## Dataset

### Source
We use a subset of the **20 Newsgroups dataset**, specifically:

**Sports Category (2 newsgroups):**
- `rec.sport.baseball` - Baseball discussions
- `rec.sport.hockey` - Hockey discussions

**Politics Category (3 newsgroups):**
- `talk.politics.guns` - Gun control debates
- `talk.politics.mideast` - Middle Eastern politics
- `talk.politics.misc` - General political discussions

### Statistics
- **Total documents:** 4,618
- **Train-test split:** 80-20 (3,694 training, 924 testing)
- **Vocabulary size:** 500 most frequent terms
- **Preprocessing:** Lowercase normalization + tokenization (no stemming/stopwords removal)

## Implementation Details

### Feature Extraction Methods

#### 1. Bag of Words (BoW)
Simple word frequency counts. Each document becomes a vector where each element represents how many times a word appears.

```python
# Example: "The game was exciting. The game ended."
# Vector: [the:2, game:2, was:1, exciting:1, ended:1]
```

#### 2. Bigrams
Captures consecutive word pairs to preserve some context.

```python
# Example: "supreme court decision"
# Bigrams: ["supreme court", "court decision"]
```

#### 3. TF-IDF
Weights terms by importance: frequent in document but rare across corpus.

```python
# Formula: TF-IDF(t,d) = TF(t,d) × log(N / DF(t))
# Emphasizes discriminative terms
```

### Classification Algorithms

#### 1. Naive Bayes
Probabilistic classifier using Bayes' theorem with independence assumption.

**Key Features:**
- Laplace smoothing to handle unseen words
- Log probabilities to prevent underflow
- Extremely fast training and prediction

**Implementation:**
```python
P(class|document) ∝ P(class) × ∏ P(word|class)^count
```

#### 2. Logistic Regression
Discriminative linear classifier with gradient descent optimization.

**Key Features:**
- Learning rate: 0.1
- Iterations: 300
- Sigmoid activation with numerical stability

**Implementation:**
```python
P(class=1|x) = σ(w·x + b) = 1 / (1 + e^-(w·x + b))
```

#### 3. K-Nearest Neighbors
Instance-based learning using majority voting of k nearest training examples.

**Key Features:**
- k = 5 neighbors
- Euclidean distance metric
- No explicit training phase

## Results

### Performance Comparison

| Feature Type | Naive Bayes | Logistic Regression | KNN (k=5) |
|-------------|-------------|-------------------|-----------|
| **Bag of Words** | **96.97%** | **96.00%** | 85.50% |
| **Bigrams** | 84.52% | 83.33% | 74.13% |
| **TF-IDF** | 63.10% | 54.65% | **86.69%** |

### Key Findings

1. **Simple > Complex:** BoW outperformed bigrams across all classifiers, challenging the assumption that more complex features always improve performance.

2. **Algorithm-Feature Interaction:** 
   - Naive Bayes + BoW: Best overall (96.97%)
   - KNN + TF-IDF: KNN's best performance (86.69%)
   - Naive Bayes + TF-IDF: Catastrophic failure (63.10%)

3. **TF-IDF Surprises:** While KNN thrived with TF-IDF, both Naive Bayes and Logistic Regression performed poorly. This highlights the importance of matching features to algorithm assumptions.

4. **Computational Efficiency:** Naive Bayes offers the best speed-accuracy tradeoff, with single-pass training and fast predictions.

## Getting Started

### Prerequisites
```bash
Python 3.7+
No external ML libraries (numpy, scikit-learn) required!
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sports-politics-classifier.git
cd sports-politics-classifier
```

2. Download the 20 Newsgroups dataset:
```bash
# Download from http://qwone.com/~jason/20Newsgroups/
# Extract to project directory as '20news-18828/'
```

### Usage

Run the classifier:
```bash
python3 B23CS1095_prob4.py
```

Expected output:
```
Loading dataset...
Total: 4618

Feature: BoW
Model: NaiveBayes
Accuracy: 0.9697
Model: LogReg
Accuracy: 0.96
Model: KNN
Accuracy: 0.855

Feature: Bigrams
Model: NaiveBayes
Accuracy: 0.8452
...
```

## Project Structure

```
sports-politics-classifier/
│
├── B23CS1095_prob4.py           # Main implementation
├── README.md                    # This file
├── B23CS1095_Report_Final.pdf   # Detailed analysis report


```

## Code Architecture

### Class: TextFeatureExtractor
Handles all feature extraction (BoW, Bigrams, TF-IDF).

**Methods:**
- `tokenize(text)` - Converts text to lowercase tokens
- `get_ngrams(tokens, n)` - Generates n-grams
- `fit(documents)` - Builds vocabulary from training data
- `transform(documents)` - Converts documents to feature vectors
- `fit_transform(documents)` - Combined fit and transform

### Class: NaiveBayesClassifier
Implements multinomial Naive Bayes with Laplace smoothing.

**Methods:**
- `train(X, y)` - Estimates class priors and feature probabilities
- `predict(X)` - Returns predicted classes using maximum likelihood

### Class: LogisticRegressionClassifier
Binary logistic regression with gradient descent.

**Methods:**
- `train(X, y)` - Optimizes weights via gradient descent
- `predict(X)` - Returns predicted classes using sigmoid threshold
- `sigmoid(z)` - Numerically stable sigmoid function

### Class: KNNClassifier
K-nearest neighbors with Euclidean distance.

**Methods:**
- `train(X, y)` - Stores training data
- `predict(X)` - Finds k nearest neighbors and returns majority vote
- `dist(a, b)` - Computes Euclidean distance

## Analysis & Insights

### Why BoW Outperforms Bigrams

Despite being simpler, BoW performs better because:

1. **Less Sparsity:** With a 500-feature limit, unigrams cover far more of the vocabulary than bigrams
2. **Better Generalization:** Individual words appear more frequently than specific word pairs
3. **Statistical Robustness:** More observations per feature leads to better probability estimates

### The TF-IDF Puzzle

TF-IDF's poor performance with Naive Bayes stems from violated assumptions:

- **Naive Bayes expects counts:** The algorithm models P(word|class) as a multinomial distribution over count data
- **TF-IDF produces normalized values:** Real-valued, scaled features break the probabilistic interpretation
- **Log-probability calculations fail:** log(TF-IDF value) is meaningless in the Naive Bayes framework

KNN succeeds with TF-IDF because it only relies on distances, not distributional assumptions.

## Limitations

### Current System
- Fixed vocabulary size (500 features) limits expressiveness
- Binary classification only (sports vs politics)
- No hyperparameter tuning
- Single evaluation metric (accuracy)
- No stopword removal or stemming
- Fixed 80-20 train-test split (no cross-validation)

### Domain Limitations
- Trained on 1990s newsgroup text
- May not generalize to modern social media or news articles
- Domain-specific to sports/politics distinction

