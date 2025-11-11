import os
import asyncio
import copy
import argparse
import librosa
import math
import numpy as np
import pandas as pd

from tqdm import tqdm
from plot_curve import plot2
from sos_client import kick_model, response_to_prob


def export_csv(results_orig, thresh):
    results = copy.deepcopy(results_orig)

    for i in tqdm(range(len(results))):
        result = results[i]
        texts = "".join(["是" if item > thresh else "否" for item in result["probs"]])
        result.pop("probs")
        result["predict"] = texts
        assert len(result["predict"]) == len(result["label"])
        result["right"] = "✓" if result["predict"] == result["label"] else "✗"
        if "是" in texts:
            idx = texts.index("是")
            tick = (idx + 1) * chunk
            result["predict_tick"] = str(tick)
            if result["label"][-1] == "是":
                delay = result["predict"].find("是") - result["label"].find("是")
                result["delay"] = delay
            else:
                result["delay"] = ""
            illegal = texts[idx+1:].count("否") > 0
            result["illegal"] = illegal
        else:
            result["predict_tick"] = ""
            result["delay"] = ""
            result["illegal"] = ""

    # log
    df = pd.DataFrame(results)
    cols = df.columns.tolist()
    # idx1, idx2 = cols.index('label_tick'), cols.index('predict')
    # cols[idx1], cols[idx2] = cols[idx2], cols[idx1]
    idx_label = cols.index('label_tick')
    idx_right = cols.index('right')
    cols.insert(idx_right, cols.pop(idx_label))
    df = df[cols]
    csv_fname = os.path.join(output_path, model_path.split("/")[-1] + f"_{testset_name}_results_{thresh:.1f}.csv")
    df.to_csv(csv_fname, mode='w', index=False)

    # summarize at streaming level
    label_all = "".join([item["label"] for item in results])
    predict_all = "".join([item["predict"] for item in results])
    res = list(zip(label_all, predict_all))
    TP = sum(1 for item in res if item[0] == '是' and item[1] == '是')
    FP = sum(1 for item in res if item[0] == '否' and item[1] == '是')
    TN = sum(1 for item in res if item[0] == '否' and item[1] == '否')
    FN = sum(1 for item in res if item[0] == '是' and item[1] == '否')
    assert (FP + TN > 0) and (FN + TP > 0)
    FPR = FP / (FP + TN)
    FNR = FN / (FN + TP)
    # ACCR = (TN + TP) / (TN + TP + FN + FP)
    INVALID = (len(label_all) - (TN + TP + FN + FP)) / len(label_all)
    assert INVALID == 0
    latencies = [item["latency"] for item in results]
    latencies.remove(max(latencies))
    result_dict = {
        "thresh":f"{thresh:.2f}",
        "误警fpr":f"{FPR*100:.1f}%",
        "漏检fnr":f"{FNR*100:.1f}%",
        # "准确率":f"{ACCR*100:.1f}%",
        "FP":FP,
        "TN":TN,
        "FN":FN,
        "TP":TP,
        "latency":int(np.mean(latencies)),
        }
    print(result_dict)
    df = pd.DataFrame([result_dict])
    df.to_csv(stats_streaming_fname, mode='a', header=not os.path.exists(stats_streaming_fname), index=False)

    # summarize at utterance level
    TP = sum(1 for item in results if item['label'][-1] == '是' and '是' in item['predict'])
    FP = sum(1 for item in results if item['label'][-1] == '否' and '是' in item['predict'])
    TN = sum(1 for item in results if item['label'][-1] == '否' and '是' not in item['predict'])
    FN = sum(1 for item in results if item['label'][-1] == '是' and '是' not in item['predict'])
    assert (FP + TN > 0) and (FN + TP > 0)
    FPR = FP / (FP + TN)
    FNR = FN / (FN + TP)
    # ACCR = (TN + TP) / (TN + TP + FN + FP)
    INVALID = (len(results) - (TN + TP + FN + FP)) / len(results)
    assert INVALID == 0
    TPs = [item for item in results if item['label'][-1] == '是' and '是' in item['predict']]
    delay_TPs = []
    for tp in TPs:
        delay_TPs.append(tp["predict"].find("是") - tp["label"].find("是"))
    advances = [item for item in delay_TPs if item < 0]
    delays = [item for item in delay_TPs if item > 0]

    total = sum(1 for item in results if item["illegal"] == True or item["illegal"] == False)
    illegal = sum(1 for item in results if item["illegal"] == True)

    result_dict = {
        "thresh":f"{thresh:.2f}",
        "误警fpr":f"{FPR*100:.1f}%",
        "漏检fnr":f"{FNR*100:.1f}%",
        # "准确率":f"{ACCR*100:.1f}%",
        "FP":FP,
        "TN":TN,
        "FN":FN,
        "TP":TP,
        "correct":"",
        "advance":"",
        "delay":"",
        "illegal":"",
        }
    if delay_TPs:
        result_dict["correct"] = f"{len(delay_TPs)-len(advances)-len(delays)}" + " / " + f"{(len(delay_TPs)-len(advances)-len(delays))/len(delay_TPs)*100:.1f}%"
        if advances:
            result_dict["advance"] = f"{len(advances)}" + " / " + f"{len(advances)/len(delay_TPs)*100:.1f}%"+" / "+f"{int(-1*np.mean(advances)*200)}ms"
        if delays:
            result_dict["delay"] = f"{len(delays)}" + " / " + f"{len(delays)/len(delay_TPs)*100:.1f}%" + (" / " + f"{int(np.mean(delays)*200)}ms" if delays else "")
    if total > 0:
        result_dict["illegal"] = f"{illegal}" + " / " + f"{illegal/total*100:.1f}%"

    print(result_dict)
    df = pd.DataFrame([result_dict])
    df.to_csv(stats_utterance_fname, mode='a', header=not os.path.exists(stats_utterance_fname), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--wav_path", type=str, required=True)
    parser.add_argument("--eval_file", type=str, required=True)
    parser.add_argument("--concurrence", type=int, default=1)
    parser.add_argument('--thresh', type=float, default=None)
    parser.add_argument("--output_path", type=str, default="eval/eval")
    parser.add_argument("--debug", type=int, default=None)

    args = parser.parse_args()
    model_path = args.model_path
    wav_path = args.wav_path
    eval_file = args.eval_file
    concurrence = args.concurrence
    thresh = args.thresh
    output_path = args.output_path
    debug = args.debug

    os.makedirs(output_path, exist_ok=True)

    input_audios = []
    results = []
    df = pd.read_csv(eval_file)
    chunk = 200
    baseset = "calibrated" in eval_file
    testset_name = "base" if baseset else eval_file.split("_")[-1][:-4]
    if not baseset:
        wav_path = os.path.join(wav_path, testset_name)
    for row_idx, row in enumerate(tqdm(df.itertuples(index=False))):
        row_data = row._asdict()
        base, ext = os.path.splitext(row_data["filename"])
        sos_final = row_data["sos_final"]
        if baseset:
            base += "_interrupter"
            base += "_true" if sos_final else "_false"
        file = os.path.join(wav_path, base + ext)
        dur = int(librosa.get_duration(filename=file) * 1000)
        num_chunk = math.ceil(dur/chunk)
        if not sos_final:
            num_neg = num_chunk
            label_tick = ""
        else:
            sos_tick = int(row_data["sos_tick"])
            assert sos_tick > 0 and sos_tick % chunk
            num_neg = sos_tick // chunk
            label_tick = sos_tick
        sr = 16000
        y, _ = librosa.load(file, sr=sr, mono=True)
        for chunk_idx in range(num_chunk):
            y_chunk = y[:(chunk_idx + 1) * chunk * 16]
            y_chunk_pcm16 = (y_chunk * 32768).clip(-32768, 32767).astype(np.int16)
            input_audios.append(((str(row_idx) + "_" + str(chunk_idx)), copy.deepcopy(y_chunk_pcm16)))
        label = "否" * num_neg + "是" * (num_chunk - num_neg)
        results.append({
            "file": file,
            "label": label,
            "label_tick": str(label_tick),
        })
        if debug and row_idx > debug:
            break

    model_outputs = asyncio.run(kick_model(input_audios, concurrence))
    for idx, res in enumerate(results):
        prefix = str(idx) + "_"
        model_output = [item for item in model_outputs if item.identity.startswith(prefix)]
        model_output = sorted(model_output, key=lambda x: int(x.identity))
        assert len(model_output) == len(res["label"])
        probs = [response_to_prob(item.response) for item in model_output]
        res["probs"] = probs
        res["latency"] = int(np.mean([item.latency for item in model_output]) * 1000)
    results = [{'file': item['file'], 'latency': item['latency'], **{k: v for k, v in item.items() if k not in ['file', 'latency']}} for item in results]


    stats_streaming_fname = os.path.join(output_path, model_path.split("/")[-1] + f"_{testset_name}_stats_streaming.csv")
    stats_utterance_fname = os.path.join(output_path, model_path.split("/")[-1] + f"_{testset_name}_stats_utterance.csv")
    if thresh is not None:
        export_csv(results, thresh)
    else:
        for thresh in np.arange(0.0,1.1,0.1):
            export_csv(results, thresh)

    plot2(stats_streaming_fname)
    print("done")
