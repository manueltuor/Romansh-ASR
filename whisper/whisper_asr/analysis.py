import pandas as pd
import matplotlib.pyplot as plt

def plot_corpus_stats(duration_df: pd.DataFrame, count_df: pd.DataFrame, word_df: pd.DataFrame):
  """Generates and displays bar charts for corpus statistics."""
  _, axes = plt.subplots(1, 3, figsize=(18, 6))

  duration_df.plot(kind='bar', ax=axes[0])
  axes[0].set_title('Audio Duration by Idiom (hours)')
  axes[0].set_ylabel('Hours')
  axes[0].set_xlabel('Idiom')
  axes[0].tick_params(axis='x', rotation=45)

  count_df.plot(kind='bar', ax=axes[1])
  axes[1].set_title('Number of Utterances by Idiom')
  axes[1].set_ylabel('Count')
  axes[1].set_xlabel('Idiom')
  axes[1].tick_params(axis='x', rotation=45)

  word_df.plot(kind='bar', ax=axes[2])
  axes[2].set_title('Word Count by Idiom')
  axes[2].set_ylabel('Count')
  axes[2].set_xlabel('Idiom')
  axes[2].tick_params(axis='x', rotation=45)

  plt.tight_layout()
  plt.show()