import pyarrow.dataset as pa_ds
import pyarrow as pa
import polars as pl
from pathlib import Path
import sys

scripts_dir = Path(__file__).resolve().parent
submodule_root = scripts_dir.parent / 'omnilingual_asr'
sys.path.insert(0, str(submodule_root))

from omnilingual_asr.constants import PARQUET_DATA_ROOT, LANG_DIST_FILE_ROOT

unified_schema = pa.schema([
    ("language", pa.string()),
    ("corpus", pa.string()),
    ("audio_size", pa.int64()),
])

ds = pa_ds.dataset(
    PARQUET_DATA_ROOT,
    partitioning="hive",
    schema=unified_schema,
    exclude_invalid_files=True,
)

table = ds.to_table(columns=["language", "corpus", "audio_size"])
pl_table = pl.from_arrow(table.combine_chunks())

stats = (
    pl_table.group_by(["corpus", "language"])
    .agg((pl.col("audio_size").sum() / 3600 / 16_000).alias("hours"))
)
stats.write_csv(
    LANG_DIST_FILE_ROOT,
    separator="\t"
)
print("Stats written.")