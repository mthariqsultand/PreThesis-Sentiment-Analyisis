import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('hasil_sentimen.predictions.csv')

sentiment_count = df['sentiment'].value_counts()

# Urutan kategori
order = ['negative', 'neutral', 'positive']
sentiment_count = sentiment_count.reindex(order)

total = sentiment_count.sum()

# Warna sesuai permintaan:
# negative = merah, neutral = kuning, positive = hijau
colors = ['red', 'yellow', 'green']

plt.figure(figsize=(10, 6))

bars = plt.bar(
    sentiment_count.index,
    sentiment_count.values,
    color=colors
)

# Tambahkan ruang atas agar label tidak keluar grafik
plt.ylim(0, sentiment_count.max() * 1.35)

for bar in bars:
    height = bar.get_height()
    percentage = (height / total) * 100

    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + sentiment_count.max() * 0.03,
        f'{int(height)}\n({percentage:.1f}%)',
        ha='center',
        va='bottom',
        fontsize=15,
        fontweight='bold'
    )

plt.title(
    'Sentiment Analysis Distribution',
    fontsize=22,
    fontweight='bold',
    pad=30
)

plt.xlabel('Sentiment Category', fontsize=16, fontweight='bold')
plt.ylabel('Number of Data', fontsize=16, fontweight='bold')

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.grid(axis='y', linestyle='--', alpha=0.3)

plt.savefig(
    'sentiment_analysis_distribution.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()