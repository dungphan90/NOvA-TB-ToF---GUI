import numpy as np


def CheckInCoincidenceWindow(hitStartUpstream,
                             hitListDownstream,
                             coincidenceWindowLowerLim,
                             coincidenceWindowUpperLim):
    coincidenceList = []

    for i_ds in range(0, np.size(hitListDownstream)):
        timeDiff = hitListDownstream[i_ds] - hitStartUpstream
        if (timeDiff - coincidenceWindowLowerLim >= 0) and (timeDiff - coincidenceWindowUpperLim <= 0):
            coincidenceList = np.append(coincidenceList, hitListDownstream[i_ds])

    return coincidenceList


def TimeMatching(hitListUpstream,
                 hitListDownstream,
                 coincidenceWindowLowerLim,
                 coincidenceWindowUpperLim):
    matchedHitList = np.zeros([1, 2])
    for i_us in range(0, np.size(hitListUpstream)):
        coincidenceList = CheckInCoincidenceWindow(hitStartUpstream=hitListUpstream[i_us],
                                                   hitListDownstream=hitListDownstream,
                                                   coincidenceWindowLowerLim=coincidenceWindowLowerLim,
                                                   coincidenceWindowUpperLim=coincidenceWindowUpperLim)
        for i_ds in range(0, np.size(coincidenceList)):
            matchedHitList = np.append(matchedHitList, [[(hitListUpstream[i_us]), (coincidenceList[i_ds])]],
                                       axis=0)

    return matchedHitList