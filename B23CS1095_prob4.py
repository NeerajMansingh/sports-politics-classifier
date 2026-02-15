# Sports vs Politics classifier
# Local 20 Newsgroups dataset

import re
import math
import random
import os
from collections import Counter


class TextFeatureExtractor:
    # extract BoW, TF-IDF, n-grams
    def __init__(self, feature_type='bow', ngram_range=(1, 1), max_features=500):
        self.feature_type = feature_type
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.vocabulary = {}
        self.idf_scores = {}
        self.fitted = False
    
    def tokenize(self, text):
        # lowercase and split
        text = text.lower()
        return re.findall(r'\b\w+\b', text)
    
    def get_ngrams(self, tokens, n):
        return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    def get_all_ngrams(self, tokens):
        terms = []
        for n in range(self.ngram_range[0], self.ngram_range[1]+1):
            terms.extend(self.get_ngrams(tokens, n))
        return terms
    
    def fit(self, documents):
        # build vocab
        term_doc_count = Counter()
        all_terms = []
        
        for doc in documents:
            tokens = self.tokenize(doc)
            terms = self.get_all_ngrams(tokens) if self.feature_type == 'ngram' else tokens
            
            for term in set(terms):
                term_doc_count[term] += 1
            all_terms.extend(terms)
        
        top_terms = [t for t,_ in Counter(all_terms).most_common(self.max_features)]
        self.vocabulary = {t:i for i,t in enumerate(top_terms)}
        
        # calc IDF
        N = len(documents)
        for term in self.vocabulary:
            df = term_doc_count[term]
            self.idf_scores[term] = math.log(N/(1+df))
        
        self.fitted = True
    
    def transform(self, documents):
        # convert to vectors
        vectors = []
        for doc in documents:
            tokens = self.tokenize(doc)
            terms = self.get_all_ngrams(tokens) if self.feature_type == 'ngram' else tokens
            
            vec = [0.0]*len(self.vocabulary)
            counts = Counter(terms)
            
            for term,count in counts.items():
                if term in self.vocabulary:
                    idx = self.vocabulary[term]
                    if self.feature_type == 'tfidf':
                        # TF-IDF score
                        tf = count/len(terms) if len(terms)>0 else 0
                        vec[idx] = tf*self.idf_scores[term]
                    else:
                        # just count
                        vec[idx] = count
            vectors.append(vec)
        return vectors
    
    def fit_transform(self, docs):
        self.fit(docs)
        return self.transform(docs)


class NaiveBayesClassifier:
    # Naive Bayes
    
    def train(self,X,y):
        # calc priors and class probs
        self.classes = list(set(y))
        class_counts = Counter(y)
        self.class_probs = {c:class_counts[c]/len(y) for c in self.classes}
        
        num_features = len(X[0])
        self.feature_probs = {}
        
        for c in self.classes:
            # Laplace smoothing
            class_docs = [X[i] for i in range(len(X)) if y[i]==c]
            sums = [0]*num_features
            for doc in class_docs:
                for i,val in enumerate(doc):
                    sums[i]+=val
            
            total = sum(sums)
            self.feature_probs[c] = [(sums[i]+1)/(total+num_features) for i in range(num_features)]
    
    def predict(self,X):
        # log probs to avoid underflow
        preds=[]
        for doc in X:
            scores={}
            for c in self.classes:
                logp=math.log(self.class_probs[c])
                for i,val in enumerate(doc):
                    if val>0:
                        logp+=val*math.log(self.feature_probs[c][i])
                scores[c]=logp
            preds.append(max(scores,key=scores.get))
        return preds


class LogisticRegressionClassifier:
    # gradient descent
    def __init__(self,lr=0.1,iters=300):
        self.lr=lr
        self.iters=iters
    
    def sigmoid(self,z):
        # clip to prevent overflow
        return 1/(1+math.exp(-max(min(z,500),-500)))
    
    def train(self,X,y):
        # init weights
        self.classes=list(set(y))
        self.map={self.classes[0]:0,self.classes[1]:1}
        yb=[self.map[i] for i in y]
        
        n=len(X)
        d=len(X[0])
        self.w=[0]*d
        self.b=0
        
        # gradient descent
        for _ in range(self.iters):
            preds=[]
            for doc in X:
                z=self.b+sum(self.w[i]*doc[i] for i in range(d))
                preds.append(self.sigmoid(z))
            
            # calc gradients
            dw=[0]*d
            db=0
            for i in range(n):
                err=preds[i]-yb[i]
                db+=err
                for j in range(d):
                    dw[j]+=err*X[i][j]
            
            # update
            for j in range(d):
                self.w[j]-=(self.lr/n)*dw[j]
            self.b-=(self.lr/n)*db
    
    def predict(self,X):
        # convert back to labels
        rev={v:k for k,v in self.map.items()}
        preds=[]
        for doc in X:
            z=self.b+sum(self.w[i]*doc[i] for i in range(len(doc)))
            p=self.sigmoid(z)
            preds.append(rev[1 if p>=0.5 else 0])
        return preds


class KNNClassifier:
    # k-nearest neighbors
    def __init__(self,k=5):
        self.k=k
    
    def train(self,X,y):
        # store data
        self.X=X
        self.y=y
    
    def dist(self,a,b):
        # Euclidean distance
        return math.sqrt(sum((a[i]-b[i])**2 for i in range(len(a))))
    
    def predict(self,X):
        # find k nearest
        preds=[]
        for doc in X:
            dists=[(self.dist(doc,self.X[i]),self.y[i]) for i in range(len(self.X))]
            dists.sort(key=lambda x:x[0])
            # vote
            votes=Counter(label for _,label in dists[:self.k])
            preds.append(votes.most_common(1)[0][0])
        return preds


def calculate_metrics(y_true,y_pred):
    # just accuracy for now
    acc=sum(1 for i in range(len(y_true)) if y_true[i]==y_pred[i])/len(y_true)
    print("Accuracy:",round(acc,4))


def train_test_split(X,y,test_size=0.2):
    # shuffle and split
    idx=list(range(len(X)))
    random.shuffle(idx)
    split=int(len(X)*(1-test_size))
    train=idx[:split]
    test=idx[split:]
    return ([X[i] for i in train],
            [X[i] for i in test],
            [y[i] for i in train],
            [y[i] for i in test])


def load_20newsgroups_local(base="20news-18828"):
    # load files from disk
    sports=["rec.sport.baseball","rec.sport.hockey"]
    politics=["talk.politics.guns","talk.politics.mideast","talk.politics.misc"]
    docs=[]
    labels=[]
    
    # sports docs
    for cat in sports:
        for f in os.listdir(os.path.join(base,cat)):
            with open(os.path.join(base,cat,f),'r',encoding='latin1') as file:
                docs.append(file.read())
                labels.append("sports")
    
    # politics docs
    for cat in politics:
        for f in os.listdir(os.path.join(base,cat)):
            with open(os.path.join(base,cat,f),'r',encoding='latin1') as file:
                docs.append(file.read())
                labels.append("politics")
    
    return docs,labels


if __name__=="__main__":
    # test all feature types and models
    print("Loading dataset...")
    docs,labels=load_20newsgroups_local()
    print("Total:",len(docs))
    
    feature_types=[("BoW","bow",(1,1)),
                   ("Bigrams","ngram",(2,2)),
                   ("TFIDF","tfidf",(1,1))]
    
    for name,ftype,ngr in feature_types:
        print("\nFeature:",name)
        # split data
        X_train_docs,X_test_docs,y_train,y_test=train_test_split(docs,labels)
        
        # extract features
        extractor=TextFeatureExtractor(ftype,ngr)
        X_train=extractor.fit_transform(X_train_docs)
        X_test=extractor.transform(X_test_docs)
        
        # test models
        models=[("NaiveBayes",NaiveBayesClassifier()),
                ("LogReg",LogisticRegressionClassifier()),
                ("KNN",KNNClassifier())]
        
        for mname,model in models:
            print("Model:",mname)
            model.train(X_train,y_train)
            preds=model.predict(X_test)
            calculate_metrics(y_test,preds)
