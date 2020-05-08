import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def detect_peak_idx(time_dat, ecg_dat, order='auto'):
    """ECGデータのピークを検出．

    Args:
        time_dat (numpy.array, float): ECGの時系列生データに対する時間情報
        ecg_dat (numpy.array, float): ECGの時系列生データ
        order (str, optional): 
            'auto'でない場合はsignal.argrelmaxの引数. 
            'auto'の場合は自動でorderを決定する．
            Defaults to 'auto'.

    Returns:
        numpy.array: ecg_dat中のピークのインデックス番号.
    """
    if order=='auto':
        order_cands = np.arange(10,300,10) # grid-search from this candidates.
        p2p_range_list, p2p_std_list = [],[]
        p2p_range_min = np.inf
        p2p_std_min = np.inf
        for odr in order_cands:
            peak_idx = signal.argrelmax(ecg_dat, order=odr)[0]
            peak_time = time_dat[peak_idx]
            peak2peak = np.diff(peak_time)
            p2p_range = peak2peak.max() - peak2peak.min()
            p2p_std = peak2peak.std()
            p2p_range_list.append(p2p_range)
            p2p_std_list.append(p2p_std)
            if p2p_range_min >= p2p_range and p2p_std_min >= p2p_std:
                p2p_range_min = p2p_range
                p2p_std_min = p2p_std
                order_best = odr
        peak_idx = signal.argrelmax(ecg_dat, order=order_best)[0]
    else:
        order = int(order)
        peak_idx = signal.argrelmax(ecg_dat, order=order)[0]

    return peak_idx


def visualize_peak(time_dat, ecg_dat, peak_idx):
    """検出したピークの可視化

    Args:
        time_dat (numpy.array, float): ECGの時系列生データに対応する時間情報
        ecg_dat (numpy.array, float): ECGの時系列生データ
        peak_idx (numpy.array, int): ピークのインデックス番号
    """
    plt.figure(figsize=(20,4))
    plt.plot(time_dat, ecg_dat, label="ECG", c="blue", alpha=0.5)
    plt.plot(time_dat[peak_idx], ecg_dat[peak_idx], "ro", label="peak_max")
    plt.xlabel("time (sec)")
    plt.ylabel("ECG")
    plt.legend(loc="upper right")
    plt.show()


def ecg2rri(time_dat, peak_idx):
    """ECGをRRIに変換（Peak to peakをRRIとしている．正確には違うかもしれない）

    Args:
        time_dat (numpy.array): ECGの時系列生データに対応する時間情報
        peak_idx (numpy.array): ピークのインデックス番号

    Returns:
        numpy.array: RRI時間およびRRI
    """
    rri = np.diff(time_dat[peak_idx])
    rri_time = np.array([np.sum(rri[:m]) for m in range(rri.size)])
    return rri_time, rri
