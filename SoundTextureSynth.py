import sys
sys.path.append("/usr/local/lib/python2.7/site-packages")

import librosa
import essentia
from essentia.standard import *

import math
import gammatone as gt
from gammatone import filters
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import hilbert, chirp, welch


path = 'original_Bubble.wav'

def feature_extract(x, fs, start, end, frame_len):
    hop = frame_len / 2
    w_hann = Windowing(type='hann')
    W = w_hann(np.ones(frame_len, dtype='single'))
    spectrum = Spectrum()

    # spectral feature
    Env = Envelope()

    # sub-band init
    central_freq = gt.filters.centre_freqs(44100, 30, 20)
    erb_filter = gt.filters.make_erb_filters(44100, central_freq, width=1.0)

    loader = essentia.standard.MonoLoader(filename=x)
    audio = loader()

    feature = np.zeros((30, 5))

    # extract sub-band
    SB = gt.filters.erb_filterbank(audio[fs * start:fs * end], erb_filter)
    SB = np.array(SB, dtype='single')

    R = np.zeros((SB.shape[1] + frame_len * 2), dtype='single')
    for a, sb in enumerate(SB):
        r = np.zeros(len(sb) + frame_len * 2)

        frame_num = 0
        SKts = []
        for frame in FrameGenerator(sb, frameSize=frame_len, hopSize=hop):

            # Envelope
            Env = np.power(sum(np.power(w_hann(frame), 2)) / sum(np.power(W, 2)), 0.5)

            # compression
            SKt = np.power(Env, 0.3)
            SKts.append(SKt)

            # residual 
            r[frame_num * hop:(frame_num * hop) + frame_len] = w_hann(frame / Env)
            frame_num += 1

        SKts = np.array(SKts, dtype='single')

        # feature computing
        m1 = sum(SKts) / frame_num
        m2 = sum(np.power(SKts - m1, 2)) / frame_num
        m3 = sum(np.power(SKts - m1, 3)) / frame_num
        m4 = sum(np.power(SKts - m1, 4)) / frame_num

        M1 = m1
        M2 = m2 / np.power(m1, 2)
        M3 = m3 / np.power(m2, 1.5)
        M4 = m4 / np.power(m2, 2)

        feature[a][0] = M1
        feature[a][1] = M2
        feature[a][2] = M3
        feature[a][3] = M4

        feature[a][4] = np.var(sb)
        # residual
        R += gt.filters.erb_filterbank(r, np.reshape(erb_filter[a], (1, 10)))[0]

    # All pole filter extract with LPC
    lpc = LPC(order=199)
    frame_num = 0
    A = np.zeros(200)
    s = Spectrum()
    for frame in FrameGenerator(R, frameSize=fs, hopSize=fs / 2):
        a, k = lpc(s(w_hann(frame)))
        A += a
        frame_num += 1

    A /= frame_num
    return feature, A


def synth(feature, itertime, sample_len, learning_rate, momentum, AR_coe):
    frame_len = 2048
    hop = frame_len / 2

    # white noise
    mean = 0
    std = 1
    num_samples = sample_len
    sample = np.random.normal(mean, std, size=num_samples)

    # gammatone filter-bank
    central_freq = gt.filters.centre_freqs(44100, 30, 20)
    erb_filter = gt.filters.make_erb_filters(44100, central_freq, width=1.0)

    # compress and Envelope
    Env = Envelope()

    # window
    w_hann = Windowing(type='hann')
    W = w_hann(np.ones(frame_len, dtype='single'))
    # learning rate
    lr = learning_rate
    # momentum
    alpha = momentum

    mom_M1 = 0
    mom_M2 = 0
    mom_M3 = 0
    mom_M4 = 0

    _iter = 0

    stactic = np.zeros((30, 5))

    SB_sample = np.array(gt.filters.erb_filterbank(sample, erb_filter), dtype='single')
    new_sample = np.zeros((30, sample_len))
    SKts_full = []
    for a, sb in enumerate(SB_sample):
        SKts = []
        frame_num = 0
        for frame in FrameGenerator(sb, frameSize=frame_len, hopSize=hop):

            Env = np.power(sum(np.power(w_hann(frame), 2)) / sum(np.power(W, 2)), 0.5)
            # compression
            SKt = np.power(Env, 0.3)

            SKts.append(SKt)

        SKts = np.array(SKts, dtype='single')
        SKts_full.append(SKts)

    SKts_full = np.array(SKts_full, dtype='single')

    print SKts_full.shape
    frame_num = SKts_full.shape[1]
    while (_iter < itertime):
        for a in xrange(30):
            # gradient decent in features
            # M1~4 caculation
            m1 = sum(SKts_full[a]) / frame_num
            m2 = sum(np.power(SKts_full[a] - m1, 2, dtype=np.float128)) / frame_num
            m3 = sum(np.power(SKts_full[a] - m1, 3, dtype=np.float128)) / frame_num
            m4 = sum(np.power(SKts_full[a] - m1, 4, dtype=np.float128)) / frame_num

            M1 = m1
            M2 = m2 / np.power(m1, 2)
            M3 = m3 / np.power(m2, 1.5)
            M4 = m4 / np.power(m2, 2)

            stactic[a][0] = M1
            stactic[a][1] = M2
            stactic[a][2] = M3
            stactic[a][3] = M4

            # errors
            err_M1 = (feature[a][0] - M1)
            err_M2 = (feature[a][1] - M2)
            err_M3 = (feature[a][2] - M3)
            err_M4 = (feature[a][3] - M4)

            # update in each envelope
            for b in xrange(SKts_full.shape[1]):
                # M2 update
                delta_2 = (1.0 / frame_num) * 2 * (SKts_full[a][b] - M1)
                #                       why not m1^2
                deri_M2 = (err_M2) * (delta_2 * np.power(m1, 2) - m2 * 2 * m1 * (1 / frame_num)) / np.power(m1, 4,
                                                                                                            dtype=np.float128)
                SKts_full[a][b] += lr * ((alpha * deri_M2) + (1 - alpha) * mom_M2)
                mom_M2 = deri_M2

                # M3 update
                delta_3 = (1.0 / frame_num) * 3 * np.power((SKts_full[a][b] - M1), 2)
                deri_M3 = (err_M3) * (delta_3 * np.power(m2, 1.5) - m3 * 1.5 * np.power(m2, 0.5) * delta_2) / np.power(
                    m2, 3)
                SKts_full[a][b] += lr / 10.0 * ((alpha * deri_M3) + (1 - alpha) * mom_M3)
                mom_M3 = deri_M3

                # M4 update
                delta_4 = (1.0 / frame_num) * 4 * np.power((SKts_full[a][b] - M1), 3)
                deri_M4 = (err_M4) * (delta_4 * np.power(m2, 2) - m4 * 2 * m2 * delta_2) / np.power(m2, 4)
                SKts_full[a][b] += lr / 100.0 * ((alpha * deri_M4) + (1 - alpha) * mom_M4)
                mom_M4 = deri_M4

            # mean adjustment
            SKts_full[a] = SKts_full[a] * (feature[a][0] / M1)

            error = sum(np.power(feature - stactic, 2))

        _iter += 1

    # residual synthesize

    A = np.array(AR_coe, dtype='single')
    B = np.ones(1, dtype='single')
    h = IIR(denominator=A, numerator=B)
    residual = h(np.array(sample, dtype='single'))

    SB_resi = np.array(gt.filters.erb_filterbank(residual, erb_filter), dtype='single')
    # envelope combined with residual
    SB_result = np.zeros((30, sample_len + frame_len * 2))
    result = np.zeros(sample_len + frame_len * 2)
    for a, sb in enumerate(SB_resi):
        frame_num = 0
        for frame in FrameGenerator(sb, frameSize=frame_len, hopSize=hop):
            Env = np.power(sum(np.power(w_hann(frame), 2)) / sum(np.power(W, 2)), 0.5)

            # compression
            SKt = np.power(Env, 0.3)

            # new envelope combined with filtered residual
            SB_result[a][frame_num * hop:(frame_num * hop) + frame_len] = w_hann(frame / Env) * SKts_full[a][frame_num]

            frame_num += 1

        # varience adjustment
        SB_result[a] *= np.power(feature[a][4] / np.var(SB_result[a]), 0.5)

        # subband combined
        result += gt.filters.erb_filterbank(SB_result[a], np.reshape(erb_filter[a], (1, 10)))[0]

    return result, stactic