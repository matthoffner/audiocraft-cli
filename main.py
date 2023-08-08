import torchaudio
import datetime
import time
import argparse

from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

parser = argparse.ArgumentParser(description="Generate music with a specified duration.")
parser.add_argument("--duration", type=int, default=2, help="Duration of the generated music in seconds.")
args = parser.parse_args()
duration = args.duration

start_time = time.time()
current_date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

model = MusicGen.get_pretrained('facebook/musicgen-small')
model.set_generation_params(
    use_sampling=True,
    top_k=250, #int
    top_p=1, #float
    temperature=0.8, #float
    cfg_coef=9.0,  #float
    #extend_stride=20, 
    duration=duration # generate x seconds
)  
descriptions = [
    ''
]

wav = model.generate(descriptions)  # generates as many samples as in comma separated descriptions

for idx, one_wav in enumerate(wav):
    file_name = f'{idx}_{current_date_time}'

    audio_write(
        file_name,
        one_wav.cpu(),
        format = "wav",
        sample_rate = model.sample_rate,
        normalize = True,
        strategy = "loudness", #clip, peak, rms, loudness
        loudness_headroom_db = 16,
        loudness_compressor = True,
        log_clipping = True #only when strategy = loudness
    )

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Generated {descriptions}, duration: {duration}s, elapsed time: {elapsed_time:.2f}s")
