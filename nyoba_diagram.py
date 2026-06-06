import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('hasil_sentimen.predictions.csv')

# Hitung jumlah sentimen
sentiment_count = df['sentiment'].value_counts()

total = sentiment_count.sum()


plt.figure(figsize=(8,5))
bars = plt.bar(sentiment_count.index, sentiment_count.values)

for bar in bars:
    height = bar.get_height()
    percentage = (height / total) * 100

    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f'{int(height)}\n({percentage:.1f}%)',
        ha='center'
    )

plt.title('Sentiment Analysis Distribution')
plt.xlabel('Sentiment Category')
plt.ylabel('Number of Data')

plt.savefig('sentiment_analysis_distribution.png', dpi=300, bbox_inches='tight')
plt.show()