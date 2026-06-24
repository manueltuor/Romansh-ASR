import subprocess
import soundfile as sf
import unicodedata
import re

def get_best_gpu():
  """
  Query system GPUs and return the index of the device with the most free VRAM.
   Defaults to GPU 0 if command fails or no GPUs are found.
  """
  try:
    # Execute system query directly to retrieve a clean comma-separated list of live stats
    cmd = ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits']
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
    gpus = []
    for line in result.stdout.strip().split('\n'):
      if line.strip():
        idx, free_mem = [int(x.strip()) for x in line.split(',')]
        gpus.append((idx, free_mem))
        
    if not gpus:
      return 0
        
    max_free = max(free for _, free in gpus)
    best_gpus = [idx for idx, free in gpus if free == max_free]
    selected = best_gpus[len(best_gpus)-1]
        
    print(f"Selected GPU {selected} with {max_free} MiB free memory")
    return selected
        
  except Exception as e:
    print(f"Error detecting best GPU: {e}")
    print("Defaulting to GPU 0")
    return 0

def get_idiom_name_by_folder(folder_name):
  """
  Map raw dataset subfolder naming conventions to formal idiom names.
  """
  name = (folder_name.split("-")[0])[2:]
  match name:
    case "sursilv":
      return "Sursilvan"
    case "sursiv":
      return "Surmiran"
    case "sutsilv":
      return "Sutsilvan"
    case "puter":
      return "Puter"
    case "vallader":
      return "Vallader"
    case _:
      return "RG"

def get_audio_duration(path):
  """
  Calculate the total duration of an audio file in seconds.
  
  Reads file metadata headers via soundfile to prevent loading heavy 
  raw audio sample arrays completely into memory.
  """
  try:
    with sf.SoundFile(path) as f:
      # Total sample frames divided by sample rate yields true duration float
      return len(f) / f.samplerate
  except Exception as e:
    print(f"Could not read {path}: {e}")
    return 0.0
  
def normalize_romansh_text(text: str) -> str:
    """Normalize text for Romansh ASR:
    - Unicode NFD → remove combining characters → NFC
    - Lowercase
    - Remove punctuation (keep letters and whitespace)
    - Collapse multiple spaces
    """
    if not isinstance(text, str):
        return ""
    # Decompose character accents
    text = unicodedata.normalize('NFD', text)
    # Filter out and drop the isolated combining diacritic marks
    text = ''.join(c for c in text if not unicodedata.combining(c))
    # Re-compose structural components back into canonical Unicode representations
    text = unicodedata.normalize('NFC', text)
    # lowercase
    text = text.lower()
    # Strip away all punctuation and symbols while retaining letters and whitespace
    text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
    # Replace multiple spaces, tabs, or newlines with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text