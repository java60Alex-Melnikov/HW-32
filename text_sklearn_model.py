import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


class TextModel:
    def __init__(self, text: str):
        self.text_arr = text.split(".")
        self.cleaned_text = [res for s in self.text_arr if (res := s.strip())]
        self.vectorizer = TfidfVectorizer()
        self.embeddings_text = self.vectorizer.fit_transform(self.cleaned_text).toarray()
        