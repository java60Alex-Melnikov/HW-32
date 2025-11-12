import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


class TextModel:
    def __init__(self, text: str):
        self.text_arr = text.split(".")
        self.cleaned_text = [res for s in self.text_arr if (res := s.strip())]
        self.vectorizer = TfidfVectorizer()
        self.embeddings_text = self.vectorizer.fit_transform(self.cleaned_text).toarray()

    def getAnswers(self, question: str, nAnswers: int) -> list[str]:
        query_embedding = self.vectorizer.transform([question]).toarray()
        similarities = cosine_similarity(self.embeddings_text, query_embedding).flatten()
        sorted_indices = np.argsort(similarities)[::-1]
        result = [self.cleaned_text[i] for i in sorted_indices[:nAnswers] if similarities[i] > 0]
        return result