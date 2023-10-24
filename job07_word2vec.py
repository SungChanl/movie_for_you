import pandas as pd
from gensim.models import Word2Vec

df_review = pd.read_csv('./crawling_data/cleaned_one_review.csv')
df_review.info()

reviews = list(df_review['review'])
print(reviews[0])

tokens = []
for sentence in reviews:
    token = sentence.split()
    tokens.append(token)
print(tokens[0])
# window : 한번에 보는 단어의 수 = kernal_size / min_count : 최소 20번은 출연해야 그 단어를 사용 / workers : 논리프로세스 사용 갯수
embedding_model = Word2Vec(tokens, vector_size=100, window=4, min_count=20, workers=8, epochs=100, sg=1)
