import numpy as np
from scipy import interpolate
import pyper
r = pyper.R("C:/Program Files/R/R-3.6.3/bin/R.exe")

def Lorenzplot_analysis(rri):
    """ローレンツプロット解析による特徴量設計
    
    Args:
        rri (1d numpy array): rri (msec)

    Returns:
        tuple (float): CSI, CVI 
    """
    rri = np.stack((rri[:rri.shape[0]-1], rri[1:])).T

    # 回転行列の定義
    rotation_matrix = np.array([[np.cos(-45*np.pi/180), -np.sin(-45*np.pi/180)],
                                [np.sin(-45*np.pi/180), np.cos(-45*np.pi/180)]])
    
    # rriを原点中心に-45°回転
    rri_rot = rotation_matrix.dot(rri.T).T
    L,T = rri_rot.std(axis=0, ddof=1)
    CSI = L / T
    CVI = np.log10(16 * L * T)
    return CSI, CVI

def time_domain_analysis(rri):
    """時間ドメイン解析
    
    Args:
        rri (1d numpy array): rri (msec)
    
    Returns:
        tuple (float): rri_mean, rrsd, rmsd
    """
    rri_mean = np.nanmean(rri)
    rrsd = np.nanstd(rri)
    rmssd = np.sqrt(np.mean(np.diff(rri)**2))
    rrcv = rrsd / rri_mean
    return rri_mean, rrsd, rmssd, rrcv

def spectrum_analysis(rri_t, rri):
    """周波数解析
    
    Args:
        rri_t (1d numpy array): 開始を0とした経過時間 (sec)
        rri (1d numpy array): rri (msec)
    
    Returns:
        tuple (float): rLF, rHR, LF/HF
    """
    f = interpolate.interp1d(rri_t, rri, kind="linear")
    Hz = 1
    # ds = 0
    tf = int(rri_t.max())
    num = tf * Hz + 1
    rri_t_resample = np.linspace(0, tf, num)
    rri_resample = f(rri_t_resample)

    # 周波数解析
    r.assign("rri", rri_resample)
    r("spec = spec.pgram(ts(rri, frequency=1), spans=c(3,3), log='no')")
    freq = r.get("spec$freq")
    spec = r.get("spec$spec")
    spec_LF = spec[(freq>=0.05) & (freq<0.15)]
    spec_HF = spec[(freq>=0.15) & (freq<0.40)]
    dif = np.diff(freq)[0]
    power_LF = np.sum([(spec_LF[m] + spec_LF[m+1])*dif/2 for m in range(spec_LF.shape[0]-1)])
    power_HF = np.sum([(spec_HF[m] + spec_HF[m+1])*dif/2 for m in range(spec_HF.shape[0]-1)])
    LFHF = power_LF / power_HF
    rLF = power_LF / (power_LF + power_HF)
    rHF = power_HF / (power_LF + power_HF)
    return rLF, rHF, LFHF
    
