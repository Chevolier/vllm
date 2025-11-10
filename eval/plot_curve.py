import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['误警fpr'] = df['误警fpr'].str.rstrip('%').astype(float) / 100
    df['漏检fnr'] = df['漏检fnr'].str.rstrip('%').astype(float) / 100

    df['FPR'] = df['误警fpr'] # 假正率
    df['TPR'] = 1 - df['漏检fnr'] # 召回率（真正率）
    df['Precision'] = np.where((df['TP'] + df['FP']) == 0, 1, df['TP'] / (df['TP'] + df['FP'])) # 精确率

    return df

def plot_roc(ax, df, testset):
    roc_df = df.sort_values('FPR')
    roc_df = pd.concat([roc_df[roc_df['thresh'] == 1.0],
                    roc_df[roc_df['thresh'] != 1.0]])
    auc = np.trapz(roc_df['TPR'], roc_df['FPR'])

    ax.plot(roc_df['FPR'], roc_df['TPR'],
             marker='o',
             color='#E74C3C',
             linewidth=2,
             label=f'ROC Curve (AUC = {auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.7)
    for i, row in roc_df.iterrows():
        if row['thresh'] not in [0, 1]:
            ax.annotate(f"θ={row['thresh']}",
                         (row['FPR'], row['TPR']),
                         textcoords="offset points",
                         xytext=(10,-10),
                         ha='left')

    ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
    ax.set_ylabel('True Positive Rate (TPR/Recall)', fontsize=12)
    ax.set_title(f'ROC_{testset}', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.2)

def plot_pr(ax, df, testset):
    pr_df = df.sort_values('TPR')
    pr_df = pd.concat([pr_df[pr_df['thresh'] != 0],
                       pr_df[pr_df['thresh'] == 0]])
    auprc = np.trapz(pr_df['Precision'], pr_df['TPR'])

    ax.plot(pr_df['TPR'], pr_df['Precision'],
             marker='s',
             color='#3498DB',
             linewidth=2,
             label=f'PR Curve (AUPRC = {auprc:.3f})')

    for i, row in pr_df.iterrows():
        if row['thresh'] not in [0, 1]:
            ax.annotate(f"θ={row['thresh']}",
                         (row['TPR'], row['Precision']),
                         textcoords="offset points",
                         xytext=(5,10),
                         ha='left')

    ax.set_xlabel('Recall (TPR)', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'PR_{testset}', fontsize=14)
    ax.legend(loc='lower left')
    ax.grid(alpha=0.2)

def plot(filepath):
    dir, fname = filepath.split("/")
    files = glob.glob(os.path.join(dir, f"{fname}*"))
    files.sort()
    assert len(files) == 2
    base, _ = os.path.splitext(files[0])
    begin_idx = base.find("inference_") + len("inference_")
    end_idx = base.find("_testset") + len("_testset") + 1
    fname = "model_" + base[begin_idx:end_idx]

    fig, axes = plt.subplots(2, 2, figsize=(20, 8))
    fig.suptitle(f'{fname}', fontsize=16)

    data_chunkwise = load_data(files[0])
    plot_roc(axes[0,0], data_chunkwise, True)
    plot_pr(axes[0,1], data_chunkwise, True)

    data_utterwise = load_data(files[1])
    plot_roc(axes[1,0], data_utterwise, False)
    plot_pr(axes[1,1], data_utterwise, False)

    plt.tight_layout()
    plt.savefig(os.path.join(dir, f'{fname}.png'), dpi=300)
    plt.show()

def plot2(filepath):
    dir, fname = os.path.split(filepath)
    wildcard = os.path.join(dir, f"{fname[:-27]}*streaming*")
    print(f"plot2 wildcard {wildcard}")
    files = glob.glob(wildcard)
    files.sort()
    if len(files) < 5:
        return
    assert len(files) == 5
    base, _ = os.path.splitext(files[0])
    begin_idx = base.find("inference_") + len("inference_")
    end_idx = begin_idx + 6
    if base[end_idx] != "_":
        end_idx -= 1
    model_step = base[begin_idx:end_idx].split("_")
    if len(model_step[1]) == 3:
        model_step[1] = "0" + model_step[1]
    fname = "model_" + "_".join(model_step)

    fig, axes = plt.subplots(len(files), 2, figsize=(20, 20))
    fig.suptitle(f'{fname}', fontsize=16)

    sub_titles = [item.split("_")[-3] for item in files]
    for i in range(len(files)):
        data = load_data(files[i])
        plot_roc(axes[i,0], data, sub_titles[i])
        plot_pr(axes[i,1], data, sub_titles[i])

    plt.tight_layout()
    plt.savefig(os.path.join(dir, f'{fname}.png'), dpi=300)
    plt.show()

# plot2("eval/finetuned_hf_for_inference_4_3000_robo_stats_streaming.csv")