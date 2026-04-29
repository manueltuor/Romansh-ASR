import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
import seaborn as sns
from jiwer import wer

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

def plot_wer_comparison(
    whisper_summary: pd.DataFrame,
    omnilingual_summary: Optional[pd.DataFrame] = None,
    title: str = "WER by Idiom",
    save_path: Optional[str] = None
) -> None:
    """
    Create a bar chart of WER per idiom.

    If only `whisper_summary` is provided, plots a single‑model chart.
    If both DataFrames are provided, shows a grouped bar chart for comparison.

    Args:
        whisper_summary: Summary DataFrame from compute_wer_summary for Whisper.
        omnilingual_summary: Optional summary DataFrame for Omnilingual.
        title: Plot title.
        save_path: If provided, save the figure to this path.
    """
    plot_data = whisper_summary[whisper_summary['idiom'] != 'OVERALL'].copy()

    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(plot_data))
    width = 0.35

    if omnilingual_summary is not None:
        merged = plot_data.merge(
            omnilingual_summary[omnilingual_summary['idiom'] != 'OVERALL'],
            on='idiom',
            suffixes=('_whisper', '_omnilingual')
        )
        bars1 = ax.bar([i - width/2 for i in x], merged['wer_mean_whisper'],
                       width, label='Whisper', color='steelblue')
        bars2 = ax.bar([i + width/2 for i in x], merged['wer_mean_omnilingual'],
                       width, label='Omnilingual', color='darkorange')
        ax.set_title(title if title != "WER by Idiom" else "WER Comparison by Idiom")
        ax.legend()
    else:
        bars = ax.bar(x, plot_data['wer_mean'], color='steelblue')
        ax.set_title(title)

        for bar, wer in zip(bars, plot_data['wer_mean']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{wer:.1f}%', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Idiom')
    ax.set_ylabel('WER (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(plot_data['idiom'])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_wer_violin(references, transcriptions, idioms, title="WER Distribution by Idiom", save_path=None):
    records = []
    for ref, hyp, idiom in zip(references, transcriptions, idioms):
        if ref and hyp:
            records.append({"idiom": idiom, "wer": wer(ref, hyp) * 100})

    df_plot = pd.DataFrame(records)
    plt.figure(figsize=(12, 6))
    sns.violinplot(
        data=df_plot,
        x='idiom',
        y='wer',
        palette='muted',
        inner='quartile',          
        cut=0,                     
        scale='width'              
    )
    #sns.violinplot(x="idiom", y="wer", data=df_plot, inner="box", palette="Set2")
    plt.title(title)
    plt.xlabel("Idiom")
    plt.ylabel("WER (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()