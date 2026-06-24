import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jiwer import wer

def plot_corpus_stats(duration_df: pd.DataFrame, count_df: pd.DataFrame, word_df: pd.DataFrame):
    """Generates and displays bar charts for corpus statistics."""
    _, axes = plt.subplots(1, 3, figsize=(18, 6))

    # audio duration per idiom
    duration_df.plot(kind='bar', ax=axes[0])
    axes[0].set_title('Audio Duration by Idiom (hours)')
    axes[0].set_ylabel('Hours')
    axes[0].set_xlabel('Idiom')
    axes[0].tick_params(axis='x', rotation=45)

    # number of utterances per idiom
    count_df.plot(kind='bar', ax=axes[1])
    axes[1].set_title('Number of Utterances by Idiom')
    axes[1].set_ylabel('Count')
    axes[1].set_xlabel('Idiom')
    axes[1].tick_params(axis='x', rotation=45)

    # number of words per idiom
    word_df.plot(kind='bar', ax=axes[2])
    axes[2].set_title('Word Count by Idiom')
    axes[2].set_ylabel('Count')
    axes[2].set_xlabel('Idiom')
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

def plot_wer_violin(references, transcriptions, idioms, title="WER Distribution by Idiom", save_path=None):
    """
    Computes sample-level Word Error Rates and visualizes the density 
    distribution per idiom using a violin plot.

    Args:
        references (list[str]): Clean ground-truth target text strings.
        transcriptions (list[str]): Model-generated transcription hypotheses.
        idioms (list[str]): Structural category labels tracking regional dialects.
        title (str): Title header applied to the generated figure.
        save_path (str | None): Optional target file path to export the plot to disk.
    """
    records = []
    # Element-wise evaluation loop mapping references against predictions
    for ref, hyp, idiom in zip(references, transcriptions, idioms):
        if ref and hyp:
            # Calculate standard WER ratio and convert it directly to a percentage scalar
            records.append({"idiom": idiom, "wer": wer(ref, hyp) * 100})

    # Convert dictionary tracking records into a DataFrame to parse with Seaborn
    df_plot = pd.DataFrame(records)
    plt.figure(figsize=(12, 6))
    # Render density curve distributions across the categorical groups
    sns.violinplot(
        data=df_plot,
        x='idiom',
        y='wer',
        palette='muted',
        inner='quartile',          
        cut=0,                     
        scale='width'              
    )
    # Configure axes layout parameters
    plt.title(title)
    plt.xlabel("Idiom")
    plt.ylabel("WER (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()