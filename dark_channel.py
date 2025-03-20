import cv2
import math
import numpy as np
from scipy.signal import convolve2d


def DarkChannel(im, sz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark


def AtmLight(im, dark):
    [h, w] = im.shape[:2]
    imsz = h * w
    numpx = int(max(math.floor(imsz / 1000), 1))
    darkvec = dark.reshape(imsz)
    imvec = im.reshape(imsz, 3)

    indices = darkvec.argsort()
    indices = indices[imsz - numpx::]

    atmsum = np.zeros([1, 3])
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx
    return A


def TransmissionEstimate(im, A, sz):
    omega = 0.95
    im3 = np.empty(im.shape, im.dtype)

    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]

    transmission = 1 - omega * DarkChannel(im3, sz)
    return transmission


def Guidedfilter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    return mean_a * im + mean_b


def TransmissionRefine(im, et):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray) / 255
    r = 60
    eps = 0.0001

    return Guidedfilter(gray, et, r, eps)


def Recover(im, t, A, tx=0.1):
    res = np.empty(im.shape, im.dtype)
    t = cv2.max(t, tx)

    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]

    return res


def estimate_beta(T, D):
    # 过滤掉深度图像中为0的区域
    mask = (D > 0) & (D < 1)
    T_filtered = T[mask]
    D_filtered = D[mask]


    # 计算透过率参数beta
    beta = -np.log(T_filtered) / D_filtered
    beta_mean = np.mean(beta)  # 取计算结果当中的平均数

    T_new = np.exp(-beta_mean * D)

    return beta_mean, T_new




def lidar_guided_filter(I, p, r, eps=1e-3):
        # 定义mask
        mask = p > 0
        p = p * mask

        # 定义窗口图f_k
        f_k = cv2.boxFilter(mask.astype(np.float32), -1, (r, r), normalize=False)
        # 计算m_p
        m_p = cv2.boxFilter(p + 1e-6, -1, (r, r), normalize=False) / (f_k + 1e-6)
        # 计算m_Ip
        m_Ip = cv2.boxFilter(I * p + 1e-6, -1, (r, r), normalize=False) / (f_k + 1e-6)
 
        # 计算m_II
        I_mask = I * mask
        m_II = cv2.boxFilter(I_mask * I_mask + 1e-6, -1, (r, r), normalize=False) / (f_k + 1e-6)

        # 其余步骤不变
        cov_Ip = m_Ip - m_p * m_p
        var_I = m_II - m_p * m_p

        a = cov_Ip / (var_I + eps)
        b = m_p - a * m_p

        m_a = cv2.boxFilter(a, -1, (r, r))
        m_b = cv2.boxFilter(b, -1, (r, r))
        cv2.imshow('m_a', m_a)
        cv2.waitKey(0)
        cv2.imshow('m_b', m_b)
        cv2.waitKey(0)
        return m_a * I + m_b



