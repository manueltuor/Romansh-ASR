"""
This script generates the language distribution file for the romansh dataset
that is required for omnilingual finetuning.
"""

import pyarrow.dataset as pa_ds
import pyarrow as pa
import polars as pl
from pathlib import Path
import sys

scripts_dir = Path(__file__).resolve().parent
submodule_root = scripts_dir.parent / 'omnilingual_asr'
sys.path.insert(0, str(submodule_root))

from omnilingual_asr.constants import PARQUET_DATA_ROOT, LANG_DIST_FILE_ROOT

# Explicitly map the targeted metadata schema. Defining this unified view 
# guarantees safe schema casting across varied parquet part files.
unified_schema = pa.schema([
    ("language", pa.string()),
    ("corpus", pa.string()),
    ("audio_size", pa.int64()),
])

# Instantiate a lazy scanning index over the parquet directory. PyArrow automatically 
# extracts "language" and "corpus" strings directly from directory layouts like
# `corpus=XYZ/language=ABC/` thanks to the "hive" partitioning configuration.
ds = pa_ds.dataset(
    PARQUET_DATA_ROOT,
    partitioning="hive",
    schema=unified_schema,
    exclude_invalid_files=True,
)

# Stream only the three required columns into memory. 
# .combine_chunks() defragments disjoint memory chunks into an optimized, 
# contiguous memory array required for rapid parsing by the Polars engine.
table = ds.to_table(columns=["language", "corpus", "audio_size"])
pl_table = pl.from_arrow(table.combine_chunks())

# Group by tracking variables and calculate the duration metrics.
# Math calculation breakdown:
#   1. Total raw sample points / 16,000Hz baseline sample rate = Total Seconds
#   2. Total Seconds / 3,600 seconds per hour = Total Hours
stats = (
    pl_table.group_by(["corpus", "language"])
    .agg((pl.col("audio_size").sum() / 3600 / 16_000).alias("hours"))
)

# Serialize the final lightweight summary distribution dataframe 
# into a tab-separated values (.tsv) manifest file for pipeline ingestion.
stats.write_csv(
    LANG_DIST_FILE_ROOT,
    separator="\t"
)
print("Stats written.")