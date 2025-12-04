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
from sos_client import kick_model, SOS_TOKEN


TWO_STAGE_JUDGEMENT = True

def sampler_roc(pdf, thresh):
    if thresh == 0.0:
        return "是"
    if thresh == 1.0:
        return "否"
    if bc_mode and TWO_STAGE_JUDGEMENT:
        lst_speech = [item[1] for item in pdf.items() if item[0]==SOS_TOKEN.SPEECH.value]
        lst_bc = [item[1] for item in pdf.items() if item[0]==SOS_TOKEN.BC.value]
        lst_noise = [item[1] for item in pdf.items() if item[0]==SOS_TOKEN.NOISE.value]
        prob_speech = lst_speech[0] if lst_speech else None
        prob_bc = lst_bc[0] if lst_bc else None
        prob_noise = lst_noise[0] if lst_noise else None
        if (prob_speech is not None and prob_speech > thresh0) or (prob_bc is None):
            return "否"
        elif prob_noise is not None:
            return "是" if prob_bc / (prob_bc + prob_noise) > thresh else "否"
        elif prob_noise is None:
            return "是"
    else:
        for token, prob in pdf.items():
            if token == token_positive:
                return "是" if prob > thresh else "否"
    # assert False
    return "否"

def export_csv(results_orig, thresh):
    results = copy.deepcopy(results_orig)

    for i in tqdm(range(len(results))):
        result = results[i]
        texts = "".join([sampler_roc(item, thresh) for item in result["probs"]])
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
    parser.add_argument("--model_type", type=str, choices=["binary", "multi"], default="binary")
    parser.add_argument("--eval_type", type=str, choices=["sos", "bc"], default="sos")
    parser.add_argument('--thresh0', type=float, default=.5)
    parser.add_argument("--wav_path", type=str, required=True)
    parser.add_argument("--eval_file", type=str, required=True)
    parser.add_argument("--concurrence", type=int, default=1)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--debug", type=int, default=None)

    args = parser.parse_args()
    model_path = args.model_path
    model_type = args.model_type
    eval_type = args.eval_type
    thresh0 = args.thresh0
    wav_path = args.wav_path
    eval_file = args.eval_file
    concurrence = args.concurrence
    output_path = args.output_path if args.output_path else f"eval/eval_{os.path.split(model_path)[-1].split('_')[-2]}_{eval_type}"
    debug = args.debug

    os.makedirs(output_path, exist_ok=True)

    baseset = "calibrated" in eval_file
    testset_name = "base" if baseset else eval_file.split("_")[-1][:-4]
    stats_streaming_fname = os.path.join(output_path, model_path.split("/")[-1] + f"_{testset_name}_stats_streaming.csv")
    stats_utterance_fname = os.path.join(output_path, model_path.split("/")[-1] + f"_{testset_name}_stats_utterance.csv")
    chunk = 200
    bc_mode = model_type == "multi" and eval_type == "bc"

    input_audios = []
    results = []
    df = pd.read_csv(eval_file)
    if not baseset:
        wav_path = os.path.join(wav_path, testset_name)
    row_idx = 0
    for row in tqdm(df.itertuples(index=False)):
        row_data = row._asdict()
        base, ext = os.path.splitext(row_data["filename"])
        bc_tick = int(row_data["bc_tick"]) if not np.isnan(row_data["bc_tick"]) else None
        sos_tick = int(row_data["sos_tick"]) if not np.isnan(row_data["sos_tick"]) else None
        assert sos_tick if row_data["sos_final"] else not sos_tick
        if bc_tick and sos_tick:
            assert bc_tick < sos_tick
        if baseset:
            base += "_interrupter"
            base += "_true" if sos_tick else "_false"
        file = os.path.join(wav_path, base + ext)
        dur = int(librosa.get_duration(filename=file) * 1000)
        num_chunk = math.ceil(dur/chunk)
        if bc_mode and sos_tick:
            num_chunk = math.floor(sos_tick/chunk)
            if num_chunk <= 0:
                continue
        tick = bc_tick if bc_mode else sos_tick
        if not tick:
            num_neg = num_chunk
            label_tick = ""
        else:
            assert tick > 0 and tick % chunk
            num_neg = tick // chunk
            label_tick = tick
            if bc_mode and (num_chunk - num_neg > 5):
                num_chunk -= (num_chunk - num_neg - 5) # avoid long tail after True

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
        row_idx += 1
        if debug is not None and row_idx >= debug:
            break

    model_outputs = asyncio.run(kick_model(model_path, input_audios, concurrence))
    for idx, res in enumerate(results):
        prefix = str(idx) + "_"
        model_output = [item for item in model_outputs if item.identity.startswith(prefix)]
        model_output = sorted(model_output, key=lambda x: int(x.identity))
        assert len(model_output) == len(res["label"])
        res["probs"] = [item.probs for item in model_output]
        res["latency"] = int(np.mean([item.latency for item in model_output]) * 1000)
    results = [{'file': item['file'], 'latency': item['latency'], **{k: v for k, v in item.items() if k not in ['file', 'latency']}} for item in results]

    if eval_type == "bc":
        for result in results:
            label_tick = int(result['label_tick']) if result['label_tick'] else 0
            if label_tick == 190:
                assert result['label'][0] == "是"
                result['label'] = "否" + result['label'][1:]

    token_positive = SOS_TOKEN.BC.value if bc_mode else (SOS_TOKEN.YES.value if model_type == "binary" else SOS_TOKEN.SPEECH.value)
    token_positive = token_positive
    for thresh in np.arange(0.0,1.1,0.1):
        export_csv(results, thresh)

    plot2(stats_streaming_fname, eval_type)
    print("done")
