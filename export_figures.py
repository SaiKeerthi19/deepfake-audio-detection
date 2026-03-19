#!/usr/bin/env python3
"""Export all figures from the Deepfake Audio Detection notebook as PNGs."""

import os, sys, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc)
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import librosa, librosa.display

plt.rcParams['figure.figsize'] = (12, 6)
LABEL_NAMES = ['bonafide', 'spoof']
OUT_DIR = '/Users/pobbamk/report_figures'
os.makedirs(OUT_DIR, exist_ok=True)

DATASET_ROOT = '/Users/pobbamk/ASVspoof2019/LA/LA'
FEAT_DIR = '/Users/pobbamk/features'
TRAIN_AUDIO = os.path.join(DATASET_ROOT, 'ASVspoof2019_LA_train', 'flac')
PROTO_DIR = os.path.join(DATASET_ROOT, 'ASVspoof2019_LA_cm_protocols')
TRAIN_PROTO = os.path.join(PROTO_DIR, 'ASVspoof2019.LA.cm.train.trn.txt')
DEV_PROTO = os.path.join(PROTO_DIR, 'ASVspoof2019.LA.cm.dev.trl.txt')
EVAL_PROTO = os.path.join(PROTO_DIR, 'ASVspoof2019.LA.cm.eval.trl.txt')
SR = 16000; DUR = 4.0; MAX_SAMP = int(SR*DUR)
N_MFCC=20; N_FFT=512; HOP=256; N_MELS=128; MEL_T=251

print("=" * 60)
print("  EXPORTING ALL FIGURES FOR REPORT")
print("=" * 60)

# ===== PARSE PROTOCOLS =====
def parse_proto(path):
    rows = []
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            if len(p)>=5:
                rows.append({'speaker':p[0],'filename':p[1],'system':p[2],
                             'key':p[4],'label':0 if p[4]=='bonafide' else 1})
    return pd.DataFrame(rows)

def parse_proto_eval(path):
    rows = []
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            if len(p)>=5:
                rows.append({'speaker':p[0],'filename':p[1],'system':p[3],
                             'key':p[4],'label':0 if p[4]=='bonafide' else 1})
    return pd.DataFrame(rows)

df_tr = parse_proto(TRAIN_PROTO)
df_dv = parse_proto(DEV_PROTO)
df_ev = parse_proto(EVAL_PROTO)
df_eval = parse_proto_eval(EVAL_PROTO)

for n,d in [('Train',df_tr),('Dev',df_dv),('Eval',df_ev)]:
    print(f'{n}: {len(d)} total, {(d.key=="bonafide").sum()} bon, {(d.key=="spoof").sum()} spf')

# ===== LOAD FEATURES =====
print("\nLoading pre-extracted features...")
def load_feat(name):
    d = np.load(os.path.join(FEAT_DIR, f'{name}_features.npz'))
    return {'hc': d['hc'], 'mel': d['mel'], 'labels': d['labels']}

train_data = load_feat('train')
dev_data = load_feat('dev')
eval_data = load_feat('eval')

for n,d in [('Train',train_data),('Dev',dev_data),('Eval',eval_data)]:
    print(f"  {n}: HC {d['hc'].shape}, Mel {d['mel'].shape}")

X_tr = np.nan_to_num(train_data['hc']); y_tr = train_data['labels']
X_dv = np.nan_to_num(dev_data['hc']);   y_dv = dev_data['labels']
X_ev = np.nan_to_num(eval_data['hc']);  y_ev = eval_data['labels']

# ===== HELPERS =====
def calc_metrics(yt, yp, yprob=None):
    m = {'acc': accuracy_score(yt,yp),
         'prec_b': precision_score(yt,yp,pos_label=0,zero_division=0),
         'rec_b':  recall_score(yt,yp,pos_label=0,zero_division=0),
         'f1_b':   f1_score(yt,yp,pos_label=0,zero_division=0),
         'prec_s': precision_score(yt,yp,pos_label=1,zero_division=0),
         'rec_s':  recall_score(yt,yp,pos_label=1,zero_division=0),
         'f1_s':   f1_score(yt,yp,pos_label=1,zero_division=0),
         'f1_mac': f1_score(yt,yp,average='macro',zero_division=0)}
    cm = confusion_matrix(yt,yp,labels=[0,1])
    m['cm']=cm; m['FP']=cm[0,1]; m['FN']=cm[1,0]; m['TP']=cm[1,1]; m['TN']=cm[0,0]
    if yprob is not None:
        fpr,tpr,_=roc_curve(yt,yprob,pos_label=1)
        m['auc']=auc(fpr,tpr); m['fpr']=fpr; m['tpr']=tpr
    return m

def compute_eer(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer

def compute_auc_score(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    return auc(fpr, tpr)

def load_audio(fp):
    try:
        a,_ = librosa.load(fp, sr=SR, mono=True)
    except:
        return np.zeros(MAX_SAMP, dtype=np.float32)
    mx = np.max(np.abs(a))
    if mx>0: a = a/mx
    if len(a)<MAX_SAMP: a = np.pad(a,(0,MAX_SAMP-len(a)))
    else: a = a[:MAX_SAMP]
    return a.astype(np.float32)

def save_cm(m, title, filename):
    fig,ax=plt.subplots(figsize=(6,5))
    sns.heatmap(m['cm'],annot=True,fmt='d',cmap='Blues',
                xticklabels=LABEL_NAMES,yticklabels=LABEL_NAMES,ax=ax,annot_kws={'size':14})
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix - {title}')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()

# ==========================================
# FIGURE 1: Class Distribution
# ==========================================
print("\n[1/12] Class Distribution...")
fig, axes = plt.subplots(1,3,figsize=(15,5))
for ax,(n,d) in zip(axes,[('Train',df_tr),('Dev',df_dv),('Eval',df_ev)]):
    c = d.key.value_counts()
    ax.bar(c.index, c.values, color=['#2ecc71','#e74c3c'], alpha=.8, edgecolor='k')
    ax.set_title(f'{n} ({len(d):,})', fontsize=14)
    for b,v in zip(ax.patches, c.values):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+50, f'{v:,}', ha='center', fontweight='bold')
plt.suptitle('ASVspoof 2019 LA - Class Distribution', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '01_class_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 01_class_distribution.png")

# ==========================================
# FIGURE 2: Bonafide vs Spoof Visualizations
# ==========================================
print("[2/12] Bonafide vs Spoof Visualizations...")
fig,axes = plt.subplots(4,3,figsize=(18,14))
for i,fn in enumerate(df_tr[df_tr.key=='bonafide'].filename.values[:3]):
    a = load_audio(os.path.join(TRAIN_AUDIO,f'{fn}.flac'))
    axes[0,i].plot(a,color='green',lw=.5); axes[0,i].set_title(f'Bonafide: {fn}',fontsize=10)
    ms = librosa.power_to_db(librosa.feature.melspectrogram(y=a,sr=SR,n_fft=N_FFT,hop_length=HOP,n_mels=N_MELS),ref=np.max)
    librosa.display.specshow(ms,sr=SR,hop_length=HOP,x_axis='time',y_axis='mel',ax=axes[1,i])
for i,fn in enumerate(df_tr[df_tr.key=='spoof'].filename.values[:3]):
    a = load_audio(os.path.join(TRAIN_AUDIO,f'{fn}.flac'))
    axes[2,i].plot(a,color='red',lw=.5); axes[2,i].set_title(f'Spoof: {fn}',fontsize=10)
    ms = librosa.power_to_db(librosa.feature.melspectrogram(y=a,sr=SR,n_fft=N_FFT,hop_length=HOP,n_mels=N_MELS),ref=np.max)
    librosa.display.specshow(ms,sr=SR,hop_length=HOP,x_axis='time',y_axis='mel',ax=axes[3,i])
plt.suptitle('Bonafide vs Spoof Audio', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '02_bonafide_vs_spoof.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 02_bonafide_vs_spoof.png")

# ==========================================
# FIGURE 3: Feature Distributions
# ==========================================
print("[3/12] Feature Distributions...")
fnames = ['MFCC_1','MFCC_2','MFCC_3','Spec_Centroid','Spec_Rolloff','ZCR']
fidx = [0,1,2,60,62,64]
fig,axes=plt.subplots(2,3,figsize=(16,8))
for ax,fi,fn in zip(axes.flat,fidx,fnames):
    ax.hist(train_data['hc'][train_data['labels']==0,fi],bins=50,alpha=.6,label='Bonafide',color='green',density=True)
    ax.hist(train_data['hc'][train_data['labels']==1,fi],bins=50,alpha=.6,label='Spoof',color='red',density=True)
    ax.set_title(fn); ax.legend(fontsize=9)
plt.suptitle('Feature Distributions', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '03_feature_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved 03_feature_distributions.png")

# ==========================================
# MODEL 1: Logistic Regression
# ==========================================
print("[4/12] Training Logistic Regression...")
t0 = time.time()
lr = SkPipeline([('sc',StandardScaler()),
    ('clf',LogisticRegression(C=1,max_iter=1000,random_state=42,class_weight='balanced'))])
lr.fit(X_tr, y_tr)
lr_m = calc_metrics(y_ev, lr.predict(X_ev), lr.predict_proba(X_ev)[:,1])
lr_time = time.time() - t0
print(f"  LR: {lr_time:.1f}s | Acc:{lr_m['acc']:.4f} F1(S):{lr_m['f1_s']:.4f} AUC:{lr_m['auc']:.4f}")
save_cm(lr_m, 'Logistic Regression', '04_cm_logistic_regression.png')

# ==========================================
# MODEL 2: SVM
# ==========================================
print("[5/12] Training SVM...")
t0 = time.time()
svm = SkPipeline([('sc',StandardScaler()),
    ('clf',SVC(C=1,kernel='rbf',gamma='scale',random_state=42,class_weight='balanced',probability=True))])
svm.fit(X_tr, y_tr)
svm_m = calc_metrics(y_ev, svm.predict(X_ev), svm.predict_proba(X_ev)[:,1])
svm_time = time.time() - t0
print(f"  SVM: {svm_time:.1f}s | Acc:{svm_m['acc']:.4f} F1(S):{svm_m['f1_s']:.4f} AUC:{svm_m['auc']:.4f}")
save_cm(svm_m, 'SVM', '05_cm_svm.png')

# ==========================================
# MODEL 3: CNN
# ==========================================
print("[6/12] Loading CNN...")

class DeepfakeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1,32,3,1,1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(32,64,3,1,1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(nn.Conv2d(64,128,3,1,1),nn.BatchNorm2d(128),nn.ReLU(),nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(nn.Conv2d(128,256,3,1,1),nn.BatchNorm2d(256),nn.ReLU(),nn.MaxPool2d(2))
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(nn.Dropout(.5),nn.Linear(256,128),nn.ReLU(),nn.Dropout(.3),nn.Linear(128,2))
    def forward(self, x):
        x=self.conv1(x); x=self.conv2(x); x=self.conv3(x); x=self.conv4(x)
        x=self.gap(x); x=x.view(x.size(0),-1); return self.fc(x)

class MelDS(Dataset):
    def __init__(self, mels, labels):
        self.mels = np.clip((mels.astype(np.float32)+80.)/80., 0, 1)
        self.labels = labels.astype(np.int64)
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return torch.from_numpy(self.mels[i]).unsqueeze(0), torch.tensor(self.labels[i])

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
model = DeepfakeCNN().to(device)
model.load_state_dict(torch.load('/Users/pobbamk/best_cnn.pth', map_location=device, weights_only=True))
model.eval()
print(f"  Device: {device}")

BS = 32
ev_dl = DataLoader(MelDS(eval_data['mel'], eval_data['labels']), batch_size=BS)
preds, trues, probs_list = [], [], []
with torch.no_grad():
    for x,y in ev_dl:
        x=x.to(device); out=model(x); p=torch.softmax(out,1)
        preds.extend(out.argmax(1).cpu().numpy())
        trues.extend(y.numpy())
        probs_list.extend(p[:,1].cpu().numpy())
cnn_m = calc_metrics(np.array(trues), np.array(preds), np.array(probs_list))
print(f"  CNN: Acc:{cnn_m['acc']:.4f} F1(S):{cnn_m['f1_s']:.4f} AUC:{cnn_m['auc']:.4f}")
save_cm(cnn_m, 'CNN', '06_cm_cnn.png')

all_m = {'Logistic Regression': lr_m, 'SVM': svm_m, 'CNN': cnn_m}
lr_probs = lr.predict_proba(X_ev)[:,1]
svm_probs = svm.predict_proba(X_ev)[:,1]
cnn_probs = np.array(probs_list)

# ==========================================
# FIGURE 7: ROC Curves
# ==========================================
print("[7/12] ROC Curves...")
fig,ax=plt.subplots(figsize=(8,6))
for name,m in all_m.items():
    if 'fpr' in m: ax.plot(m['fpr'],m['tpr'],lw=2,label=f'{name} (AUC={m["auc"]:.4f})')
ax.plot([0,1],[0,1],'k--',lw=1,label='Random')
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.set_title('ROC Curves - All Models'); ax.legend(); ax.grid(alpha=.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '07_roc_curves.png'), dpi=150, bbox_inches='tight')
plt.close()

# ==========================================
# FIGURE 8: Model Comparison
# ==========================================
print("[8/12] Model Comparison...")
metrics_to_plot = ['acc','prec_s','rec_s','f1_s']
labels = ['Accuracy','Precision(S)','Recall(S)','F1(S)']
x = np.arange(len(labels)); w = 0.25
fig,ax=plt.subplots(figsize=(12,6))
for i,(name,m) in enumerate(all_m.items()):
    vals = [m[k] for k in metrics_to_plot]
    ax.bar(x+i*w, vals, w, label=name, alpha=.85)
ax.set_xticks(x+w); ax.set_xticklabels(labels, fontsize=12)
ax.set_ylabel('Score'); ax.set_title('Model Comparison', fontsize=14)
ax.legend(fontsize=11); ax.grid(alpha=.3, axis='y'); ax.set_ylim(0,1.05)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '08_model_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

# ==========================================
# FIGURE 9: Cybersecurity Errors
# ==========================================
print("[9/12] Cybersecurity Errors...")
fig,axes=plt.subplots(1,2,figsize=(14,5))
names = list(all_m.keys())
fns = [all_m[n]['FN'] for n in names]
fps = [all_m[n]['FP'] for n in names]
axes[0].bar(names, fns, color='#e74c3c', alpha=.8, edgecolor='k')
axes[0].set_title('False Negatives (Missed Deepfakes)', fontsize=13); axes[0].set_ylabel('Count')
for b,v in zip(axes[0].patches, fns):
    axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+5, str(v), ha='center', fontweight='bold')
axes[1].bar(names, fps, color='#f39c12', alpha=.8, edgecolor='k')
axes[1].set_title('False Positives (False Alarms)', fontsize=13); axes[1].set_ylabel('Count')
for b,v in zip(axes[1].patches, fps):
    axes[1].text(b.get_x()+b.get_width()/2, b.get_height()+5, str(v), ha='center', fontweight='bold')
plt.suptitle('Error Analysis - Cybersecurity', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '09_cybersecurity_errors.png'), dpi=150, bbox_inches='tight')
plt.close()

# ==========================================
# PER-ATTACK ANALYSIS
# ==========================================
print("[10/12] Per-Attack Analysis...")
n_eval = len(y_ev)
if n_eval == len(df_eval):
    df_eval_used = df_eval.copy()
else:
    max_n = n_eval
    db = df_eval[df_eval.label == 0]; ds = df_eval[df_eval.label == 1]
    nb = min(len(db), max_n // 2); ns = min(len(ds), max_n - nb)
    df_eval_used = pd.concat([db.sample(nb, random_state=42), ds.sample(ns, random_state=42)]).reset_index(drop=True)

systems = df_eval_used['system'].values[:n_eval]
bon_idx = np.where(y_ev == 0)[0]
attack_systems = sorted([s for s in np.unique(systems) if s != '-'])

results = []
for sys_id in attack_systems:
    atk_idx = np.where(systems == sys_id)[0]
    if len(atk_idx) == 0: continue
    idx = np.concatenate([bon_idx, atk_idx])
    y_sub = y_ev[idx]
    row = {'System': sys_id, 'N_samples': len(atk_idx),
           'Known': 'Yes' if sys_id in ['A01','A02','A03','A04','A05','A06'] else 'No'}
    for mname, mprobs in [('LR', lr_probs), ('SVM', svm_probs), ('CNN', cnn_probs)]:
        p_sub = mprobs[idx]
        try:
            row[f'{mname}_AUC'] = compute_auc_score(y_sub, p_sub)
            row[f'{mname}_EER'] = compute_eer(y_sub, p_sub)
        except:
            row[f'{mname}_AUC'] = np.nan; row[f'{mname}_EER'] = np.nan
    results.append(row)
df_results = pd.DataFrame(results)

# EER Bar Chart
fig, ax = plt.subplots(figsize=(16, 7))
xr = np.arange(len(df_results)); w = 0.25
ax.bar(xr - w, df_results.LR_EER * 100, w, label='Logistic Regression', alpha=.85, color='#3498db')
ax.bar(xr,     df_results.SVM_EER * 100, w, label='SVM', alpha=.85, color='#e67e22')
ax.bar(xr + w, df_results.CNN_EER * 100, w, label='CNN', alpha=.85, color='#2ecc71')
ax.set_xticks(xr); ax.set_xticklabels(df_results.System, fontsize=11)
ax.set_ylabel('EER (%)', fontsize=12)
ax.set_title('Equal Error Rate per Attack System', fontsize=15, fontweight='bold')
ax.legend(fontsize=11); ax.grid(alpha=.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '10_eer_per_attack.png'), dpi=150, bbox_inches='tight')
plt.close()

# AUC Bar Chart
fig, ax = plt.subplots(figsize=(16, 7))
ax.bar(xr - w, df_results.LR_AUC, w, label='Logistic Regression', alpha=.85, color='#3498db')
ax.bar(xr,     df_results.SVM_AUC, w, label='SVM', alpha=.85, color='#e67e22')
ax.bar(xr + w, df_results.CNN_AUC, w, label='CNN', alpha=.85, color='#2ecc71')
ax.set_xticks(xr); ax.set_xticklabels(df_results.System, fontsize=11)
ax.set_ylabel('AUC', fontsize=12); ax.set_title('AUC per Attack System', fontsize=15, fontweight='bold')
ax.legend(fontsize=11); ax.grid(alpha=.3, axis='y'); ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '11_auc_per_attack.png'), dpi=150, bbox_inches='tight')
plt.close()

# ==========================================
# EER Heatmap
# ==========================================
print("[11/12] EER Heatmap...")
eer_matrix = df_results[['System','LR_EER','SVM_EER','CNN_EER']].set_index('System') * 100
eer_matrix.columns = ['Logistic Regression', 'SVM', 'CNN']
fig, ax = plt.subplots(figsize=(8, 12))
sns.heatmap(eer_matrix, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=ax,
            linewidths=.5, annot_kws={'size': 11})
ax.set_title('EER (%) per Attack System x Model\n(Lower = Better)', fontsize=14, fontweight='bold')
ax.set_ylabel('Attack System')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '12_eer_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()

# ==========================================
# SAVE RESULTS TEXT
# ==========================================
print("[12/12] Saving results summary...")
with open(os.path.join(OUT_DIR, 'results_summary.txt'), 'w') as f:
    f.write("=" * 75 + "\n")
    f.write("  MODEL COMPARISON TABLE\n")
    f.write("=" * 75 + "\n")
    header = f'{"Model":<22}{"Acc":>8}{"Prec(S)":>9}{"Rec(S)":>9}{"F1(S)":>9}{"FN":>7}{"FP":>7}{"AUC":>8}\n'
    f.write(header)
    f.write('-'*75 + "\n")
    for name, m in all_m.items():
        a = m.get('auc', 0)
        f.write(f'{name:<22}{m["acc"]:>8.4f}{m["prec_s"]:>9.4f}{m["rec_s"]:>9.4f}{m["f1_s"]:>9.4f}{m["FN"]:>7}{m["FP"]:>7}{a:>8.4f}\n')
    f.write('-'*75 + "\n")

    f.write("\n\nOVERALL AUC and EER:\n")
    for mname, mprobs in [('LR', lr_probs), ('SVM', svm_probs), ('CNN', cnn_probs)]:
        eer = compute_eer(y_ev, mprobs)
        auc_val = compute_auc_score(y_ev, mprobs)
        f.write(f'  {mname}: AUC={auc_val:.4f}, EER={eer:.4f} ({eer*100:.2f}%)\n')

    f.write("\n\nPER-ATTACK RESULTS:\n")
    f.write(f'{"System":<8}{"Known":>6}{"#Spf":>7}  |{"LR_AUC":>8}{"LR_EER":>8}  |{"SVM_AUC":>9}{"SVM_EER":>9}  |{"CNN_AUC":>9}{"CNN_EER":>9}\n')
    f.write('-'*90 + "\n")
    for _, r in df_results.iterrows():
        f.write(f'{r["System"]:<8}{r["Known"]:>6}{int(r["N_samples"]):>7}  |{r["LR_AUC"]:>8.4f}{r["LR_EER"]:>8.4f}  |{r["SVM_AUC"]:>9.4f}{r["SVM_EER"]:>9.4f}  |{r["CNN_AUC"]:>9.4f}{r["CNN_EER"]:>9.4f}\n')

    f.write("\n\nATTACK DIFFICULTY RANKING:\n")
    df_results['avg_EER'] = df_results[['LR_EER','SVM_EER','CNN_EER']].mean(axis=1)
    ranked = df_results.sort_values('avg_EER', ascending=False)
    for i, (_, r) in enumerate(ranked.iterrows()):
        marker = 'CRITICAL' if r.avg_EER > 0.3 else 'MODERATE' if r.avg_EER > 0.1 else 'LOW RISK'
        f.write(f'  {i+1:2d}. [{marker}] {r.System} avg EER: {r.avg_EER*100:.1f}%\n')

    f.write("\n\nCYBERSECURITY ERROR ANALYSIS:\n")
    for name, m in all_m.items():
        f.write(f'  {name}: FN={m["FN"]} (missed fakes), FP={m["FP"]} (false alarms)\n')

    f.write(f"\n\nTIMING:\n")
    f.write(f"  LR training: {lr_time:.1f}s\n")
    f.write(f"  SVM training: {svm_time:.1f}s\n")

print("\n" + "=" * 60)
print(f"  ALL FIGURES EXPORTED TO: {OUT_DIR}")
print("=" * 60)
for fn in sorted(os.listdir(OUT_DIR)):
    sz = os.path.getsize(os.path.join(OUT_DIR, fn)) / 1024
    print(f"  {fn} ({sz:.0f} KB)")
print("\nDone!")
