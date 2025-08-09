import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data
data = {
    'method': ['dense', 'hybrid', 'sparse'],
    'Hit Rate': [0.0134, 0.0134, 0],
    'MRR': [0.0056, 0.0015, 0],
    'Context Recall': [0.1839, 0.1851, 0.1747],
    'Context Relevance': [0.5165, 0.4704, 0.2271]
}

# Prepare DataFrame
df = pd.DataFrame(data)
df_melted = df.melt(id_vars='method', var_name='Metric', value_name='Score')

# Set Seaborn style and color palette
sns.set_theme(style='whitegrid', font_scale=1.1)
palette = sns.color_palette("pastel")  # Matches your uploaded plot

# Create the plot
plt.figure(figsize=(10, 6), dpi=150)
ax = sns.barplot(data=df_melted, x='Metric', y='Score', hue='method', palette=palette)

# Add value labels
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{height:.4f}', 
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=9, color='black', xytext=(0, 4),
                    textcoords='offset points')

# Styling
plt.title('Retrieval-Level Evaluation by Method')
plt.xlabel('Metric')
plt.ylabel('Score')
plt.legend(title='Method')
plt.tight_layout()

# Save high-quality version
plt.savefig("retrieval_evaluation_summary.png", dpi=300)

# Show plot
plt.show()