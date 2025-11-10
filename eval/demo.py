import asyncio
import copy
import argparse
import librosa
import math
import numpy as np

from sos_client import kick_model, response_to_prob


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-file", type=str, required=True)
    parser.add_argument("-c", "--concurrence", type=int, default=1)
    parser.add_argument("--mode", choices=["chunks", "single"], default="chunks")
    args = parser.parse_args()
    audio_file = args.audio_file
    concurrence = args.concurrence
    mode = args.mode

    input_audios = []
    sr = 16000
    y, _ = librosa.load(audio_file, sr=sr, mono=True)
    y = (y * 32768).clip(-32768, 32767).astype(np.int16)
    if mode == "single":
        input_audios.append(("0", copy.deepcopy(y)))
    else:
        chunk = 200
        dur = int(librosa.get_duration(filename=audio_file) * 1000)
        num_chunk = math.ceil(dur/chunk)
        for chunk_idx in range(num_chunk):
            y_chunk = y[:(chunk_idx + 1) * chunk * 16]
            input_audios.append(((str(chunk_idx)), copy.deepcopy(y_chunk)))

    model_outputs = asyncio.run(kick_model(input_audios, concurrence))
    model_outputs = sorted(model_outputs, key=lambda x: int(x.identity))
    probs = [response_to_prob(item.response) for item in model_outputs]
    sos_flag_list = [True if item > .5 else False for item in probs]
    latency = int(np.mean([item.latency for item in model_outputs]) * 1000)
    print(f"sos_flag_list {sos_flag_list}, latency {latency}")
