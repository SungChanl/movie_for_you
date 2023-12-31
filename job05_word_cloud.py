# 단어 시각화
import pandas as pd
from wordcloud import WordCloud
import collections
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_path = './malgun.ttf'
font_name = font_manager.FontProperties(fname=font_path).get_name()
plt.rc('font', family='NanumBarunGothic')

df = pd.read_csv('./crawling_data/cleaned_one_review.csv')
words = df.iloc[2112, 1].split()
print(words)

worddict = collections.Counter(words) # 각 단어가 몇번 나오는지 valuecounts 와 유사
worddict = dict(worddict)
print(worddict)

wordcloud_img = WordCloud(
    background_color='white', max_words=2000, font_path=font_path).generate_from_frequencies(worddict)
# 단어 출혈 빈도를 시각화
plt.figure(figsize=(12, 12))
plt.imshow(wordcloud_img, interpolation='bilinear')
plt.axis('off')
plt.show()