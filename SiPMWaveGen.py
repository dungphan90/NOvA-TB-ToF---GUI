from math import exp

import numpy as np


def peResponse(t, delay, nphotons, speAmplitude, riseTime, fallTime):
    if t < delay:
        return 0
    else:
        return -nphotons * speAmplitude * exp(-(t - delay) / fallTime) * (1 - exp(-(t - delay) / riseTime))


def waveGen(t, speAmplitude, noiseSigmaInVolt, riseTime, fallTime):
    response = np.array([0 for ti in t])
    true_response = np.array([0 for ti in t])
    nhits = 0

    while True:
        nhits = np.random.poisson(1)
        if nhits > 0:
            break

    while True:
        checkSumHits = 0
        for aHit in range(0, nhits):
            delay = np.random.uniform(t[50], t[-50])
            nphotons = np.random.poisson(3)
            checkSumHits = checkSumHits + nphotons
            true_response = true_response + np.array([peResponse(ti,
                                                                 delay,
                                                                 nphotons,
                                                                 speAmplitude,
                                                                 riseTime,
                                                                 fallTime) for ti in t])
        if checkSumHits > 0:
            break

    response = true_response + np.array([np.random.normal(0, noiseSigmaInVolt) for ti in t])

    return [response, true_response, nhits]


def getRawADC(p, res):
    return round(p / res)


def getADC(p, nBits, res, voltMin):
    guessADC = round((p - voltMin) / res)
    if guessADC < 0:
        return 0
    if guessADC > (2 ** nBits - 1):
        return 2 ** nBits - 1
    return guessADC

def digitizeWave(p, nBits, voltMin, dynamicRange, offset):
    resolution = dynamicRange / (2**nBits - 1)
    return np.array([(getADC(pi, nBits=nBits, res=resolution, voltMin=voltMin) + offset) for pi in p])

def aTrigger(dt, nsamples, speAmplitude, noiseSigmaInVolt, riseTime, fallTime):
    t = np.arange(0, nsamples*dt, dt)
    [p, true_p] = waveGen(t, speAmplitude=speAmplitude, noiseSigmaInVolt=noiseSigmaInVolt, riseTime=riseTime, fallTime=fallTime)

    return [t, p, true_p]


def aDigitizedTrigger(dt, nsamples, speAmplitude, noiseSigmaInVolt, riseTime, fallTime, nBits, voltMin, dynamicRange,
                      offset):
    nhits = -1
    t = np.arange(0, nsamples*dt, dt)
    [p, true_p, nhits] = waveGen(t, speAmplitude=speAmplitude, noiseSigmaInVolt=noiseSigmaInVolt, riseTime=riseTime,
                          fallTime=fallTime)
    digital_p = digitizeWave(p, nBits=nBits, voltMin=voltMin, dynamicRange=dynamicRange, offset=offset)
    digital_true_p = digitizeWave(true_p, nBits=nBits, voltMin=voltMin, dynamicRange=dynamicRange, offset=offset)

    return [t, digital_p, digital_true_p, nhits]