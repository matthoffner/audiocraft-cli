import datetime
import time
import argparse

from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

import sounddevice as sd

parser = argparse.ArgumentParser(description="Generate music with a specified duration.")
parser.add_argument("--duration", type=int, default=5, help="Duration of the generated music in seconds.")
parser.add_argument("--prompt", type=str, required=True, help="Prompt for music to generate.")
parser.add_argument("--use_sampling", type=bool, default=True, help="Use sampling during generation.")
parser.add_argument("--top_k", type=int, default=250, help="Value for top_k during generation.")
parser.add_argument("--top_p", type=float, default=1, help="Value for top_p during generation.")
parser.add_argument("--temperature", type=float, default=0.8, help="Value for temperature during generation.")
parser.add_argument("--cfg_coef", type=float, default=9.0, help="Value for cfg_coef during generation.")
parser.add_argument("--extend_stride", type=int, default=20, help="Stride value during extension.")
parser.add_argument("--autoplay", action="store_true", help="Automatically play the generated audio.")
parser.add_argument("--model", type=str, default='facebook/musicgen-large', help="Model name for music generation. Default is 'facebook/musicgen-large'.")
args = parser.parse_args()
duration = args.duration
prompt = args.prompt

start_time = time.time()
current_date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

model = MusicGen.get_pretrained(args.model)
model.set_generation_params(
    use_sampling=args.use_sampling,
    top_k=args.top_k,
    top_p=args.top_p,
    temperature=args.temperature,
    cfg_coef=args.cfg_coef,
    # extend_stride=args.extend_stride,
    duration=duration  # generate x seconds
)

wav = model.generate([prompt])  # generates as many samples as in comma separated descriptions

for idx, one_wav in enumerate(wav):
    file_name = f'{idx}_{current_date_time}'
    one_wav_cpu = one_wav.cpu().numpy().squeeze()

    if args.autoplay:
        sd.play(one_wav_cpu, samplerate=model.sample_rate)

    audio_write(
        file_name,
        one_wav.cpu(),
        format="wav",
        sample_rate=model.sample_rate,
        normalize=True,
        strategy="loudness",  # clip, peak, rms, loudness
        loudness_headroom_db=16,
        loudness_compressor=True,
        log_clipping=True  # only when strategy = loudness
    )

    if args.autoplay:
        sd.wait()

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Generated {prompt}, duration: {duration}s, elapsed time: {elapsed_time:.2f}s")
