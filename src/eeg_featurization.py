import numpy as np
from scipy import interpolate
import pyper
r = pyper.R("C:/Program Files/R/R-3.6.3/bin/R.exe")

def spectrum_analysis(eeg, Hz=1000):
    """脳波電位の周波数解析
    
    Args:
        eeg (1d numpy array): 脳波電位 (μV)
        Hz (int, optional): 測定時のサンプリング周波数. Defaults to 1000.
    
    Returns:
        tuple: delta, theta, alpha, beta, gammaの強度比
    """
    # 周波数解析
    r.assign("eeg", eeg)
    r.assign("hz", Hz)
    r("spec = spec.pgram(ts(eeg, frequency=hz), spans=c(3,3), log='no')")
    freq = r.get("spec$freq")
    spec = r.get("spec$spec")
    spec_delta = spec[(freq>=0.5) & (freq<4.0)]
    spec_theta = spec[(freq>=4.0) & (freq<8.0)]
    spec_alpha = spec[(freq>=8.0) & (freq<13.0)]
    spec_beta = spec[(freq>=13.0) & (freq<31.0)]
    spec_gamma = spec[(freq>=31.0) & (freq<=39.75)]
    dif = np.diff(freq)[0]
    power_delta = np.sum([(spec_delta[m] + spec_delta[m+1])*dif/2 for m in range(spec_delta.shape[0]-1)])
    power_theta = np.sum([(spec_theta[m] + spec_theta[m+1])*dif/2 for m in range(spec_theta.shape[0]-1)])
    power_alpha = np.sum([(spec_alpha[m] + spec_alpha[m+1])*dif/2 for m in range(spec_alpha.shape[0]-1)])
    power_beta = np.sum([(spec_beta[m] + spec_beta[m+1])*dif/2 for m in range(spec_beta.shape[0]-1)])
    power_gamma = np.sum([(spec_gamma[m] + spec_gamma[m+1])*dif/2 for m in range(spec_gamma.shape[0]-1)])
    power_total = power_delta + power_theta + power_alpha + power_beta + power_gamma
    rdelta = power_delta / power_total
    rtheta = power_theta / power_total
    ralpha = power_alpha / power_total
    rbeta = power_beta / power_total
    rgamma = power_gamma / power_total
    return rdelta, rtheta, ralpha, rbeta, rgamma