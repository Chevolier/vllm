import asyncio
import copy
import argparse
import librosa
import math
import numpy as np

from sos_client import kick_model, SOS_TOKEN


def sampler_binary(pdf):
    for token, prob in pdf.items():
        if token == SOS_TOKEN.YES.value:
            return "是" if prob > thresh0 else "否"
    return "否"

def sampler_multi(pdf):
    lst_speech = [item[1] for item in pdf.items() if item[0]==SOS_TOKEN.SPEECH.value]
    lst_bc = [item[1] for item in pdf.items() if item[0]==SOS_TOKEN.BC.value]
    lst_noise = [item[1] for item in pdf.items() if item[0]==SOS_TOKEN.NOISE.value]
    prob_speech = lst_speech[0] if lst_speech else None
    prob_bc = lst_bc[0] if lst_bc else None
    prob_noise = lst_noise[0] if lst_noise else None
    if prob_speech is not None and prob_speech > thresh0:
        type = "3"
    elif prob_noise is None:
        type = "2"
    elif prob_bc is None:
        type = "1"
    else:
        type = "2" if prob_bc / (prob_bc + prob_noise) > thresh1 else "1"
    print(f"prob_noise {prob_noise:.2f}, prob_bc {prob_bc:.2f}, prob_speech {prob_speech:.2f}, type {type}")
    return type

samplers = {
    "binary": sampler_binary,
    "multi": sampler_multi,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--audio-file", type=str, required=True)
    parser.add_argument("--concurrence", type=int, default=1)
    parser.add_argument('--thresh0', type=float, default=.5)
    parser.add_argument('--thresh1', type=float, default=.5)
    args = parser.parse_args()
    model_path = args.model_path
    audio_file = args.audio_file
    concurrence = args.concurrence
    thresh0 = args.thresh0
    thresh1 = args.thresh1

    model_seq = int(model_path.split('_')[-2])
    model_type = "binary" if model_seq <= 8 else "multi"
    input_audios = []
    sr = 16000
    y, _ = librosa.load(audio_file, sr=sr, mono=True)
    y = (y * 32768).clip(-32768, 32767).astype(np.int16)
    chunk = 200
    dur = int(len(y)/sr*1000)
    num_chunk = math.ceil(dur/chunk)
    for chunk_idx in range(num_chunk):
        y_chunk = y[:(chunk_idx + 1) * chunk * 16]
        input_audios.append(((str(chunk_idx)), copy.deepcopy(y_chunk)))

    model_outputs = asyncio.run(kick_model(model_path, input_audios, concurrence))
    model_outputs = sorted(model_outputs, key=lambda x: int(x.identity))
    sos_flag_list = [samplers[model_type](item.probs) for item in model_outputs]
    latency = int(np.mean([item.latency for item in model_outputs]) * 1000)
    print(f"sos_flag_list {sos_flag_list}, latency {latency}")
