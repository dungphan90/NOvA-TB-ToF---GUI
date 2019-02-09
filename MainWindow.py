import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, \
    QPushButton, QLabel
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import SiPMWaveGen as swg
import CFDHitFinder as cfd
import TimeMatcher as tm

import numpy as np

import random


class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.m = PlotCanvas(self, width=12, height=11)
        self.label = QLabel(self)
        self.left = 10
        self.top = 10
        self.title = 'NOvA TB ToF'
        self.width = 1420
        self.height = 1000

        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), Qt.white)
        self.setPalette(p)

        self.m.move(-30, -60)

        wg_button = QPushButton('WaveGen', self)
        wg_button.setToolTip('Generate SiPM waveforms')
        wg_button.setStyleSheet("background-color: red")
        wg_button.move(1150, 20)
        wg_button.resize(180, 40)

        hf_button = QPushButton('HitFinder', self)
        hf_button.setToolTip('Toggle hit starts in waveforms')
        hf_button.setStyleSheet("background-color: red")
        hf_button.move(1150, 80)
        hf_button.resize(180, 40)

        tof_button = QPushButton('TOF', self)
        tof_button.setToolTip('Toggle time-of-flight')
        tof_button.setStyleSheet("background-color: red")
        tof_button.move(1150, 140)
        tof_button.resize(180, 40)

        s0_button = QPushButton('Pedestal', self)
        s0_button.setToolTip('Pedestal and noise sigma')
        s0_button.move(1150, 200)
        s0_button.resize(180, 40)

        s1_button = QPushButton('Threshold', self)
        s1_button.setToolTip('Toggle thresholds and hit windows')
        s1_button.move(1150, 260)
        s1_button.resize(180, 40)

        s2_button = QPushButton('CFD Threshold', self)
        s2_button.setToolTip('Toggle CFD threshold')
        s2_button.move(1150, 320)
        s2_button.resize(180, 40)

        s3_button = QPushButton('Hit start-time', self)
        s3_button.setToolTip('Toggle hit start time')
        s3_button.move(1150, 380)
        s3_button.resize(180, 40)

        self.label.move(40, 20)
        self.label.resize(1000, 60)
        self.label.setText("")

        wg_button.clicked.connect(self.waveGenButtonClicked)
        hf_button.clicked.connect(self.findHitsButtonClicked)
        tof_button.clicked.connect(self.tofButtonClicked)

        s0_button.clicked.connect(self.togglePedestalClicked)
        s1_button.clicked.connect(self.toggleHitThresholdClicked)
        s2_button.clicked.connect(self.toggleCFDThresholdClicked)
        s3_button.clicked.connect(self.toggleHitStartTimeClicked)

        self.show()

    def waveGenButtonClicked(self):
        self.m.plotWave()
        self.label.setText("")

    def findHitsButtonClicked(self):
        self.m.findHits()
        self.label.setText("")

    def tofButtonClicked(self):
        self.m.showTOF()
        self.label.setText("")

    def togglePedestalClicked(self):
        self.m.togglePedestal()
        self.label.setText("")

    def toggleHitThresholdClicked(self):
        self.m.toggleHitThreshold()
        self.label.setText("")

    def toggleCFDThresholdClicked(self):
        self.m.toggleCFDThreshold()
        self.label.setText("")

    def toggleHitStartTimeClicked(self):
        self.m.toggleHitStartTime()
        self.label.setText("")


class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5.5, height=4, dpi=100):
        self.showHitLines = False
        self.hitLines1 = list()
        self.hitLines2 = list()

        self.foundHits = False

        self.showToFRegions = False
        self.ToFRegion1 = list()
        self.ToFRegion2 = list()

        self.showPedestal = False
        self.pedestalLines1 = list()
        self.pedestalLines2 = list()

        self.showHitThreshold = False
        self.hitThresholdLine1 = list()
        self.hitThresholdLine2 = list()

        self.dt = 0.2
        self.nsamples = 1024
        self.speAmplitude = 0.15
        self.noiseSigmaInVolt = 0.02
        self.riseTime = 10
        self.fallTime = 25

        self.nBits = 12
        self.voltMin = -0.8
        self.dynamicRange = 1
        self.offset = 1000

        self.cfdThreshold = 0.4
        self.nNoiseSigmaThreshold = 2.5

        self.t1 = []
        self.data1 = []
        self.true_data1 = []

        self.t2 = []
        self.data2 = []
        self.true_data2 = []

        self.hitStartIndexList1 = []
        self.hitLogic1 = []
        self.baseline1 = []
        self.noiseSigma1 = []

        self.hitStartIndexList2 = []
        self.hitLogic2 = []
        self.baseline2 = []
        self.noiseSigma2 = []

        self.matchedHitList = []

        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        self.plotWave()

    def trigGen(self):
        self.showHitLines = False
        self.foundHits = False
        self.showToFRegions = False
        self.showPedestal = False
        self.showHitThreshold = False

        [self.t1, self.data1, self.true_data1] = swg.aDigitizedTrigger(dt=self.dt,
                                                                       nsamples=self.nsamples,
                                                                       speAmplitude=self.speAmplitude,
                                                                       noiseSigmaInVolt=self.noiseSigmaInVolt,
                                                                       riseTime=self.riseTime,
                                                                       fallTime=self.fallTime,
                                                                       nBits=self.nBits,
                                                                       voltMin=self.voltMin,
                                                                       dynamicRange=self.dynamicRange,
                                                                       offset=self.offset)

        [self.t2, self.data2, self.true_data2] = swg.aDigitizedTrigger(dt=self.dt,
                                                                       nsamples=self.nsamples,
                                                                       speAmplitude=self.speAmplitude,
                                                                       noiseSigmaInVolt=self.noiseSigmaInVolt,
                                                                       riseTime=self.riseTime,
                                                                       fallTime=self.fallTime,
                                                                       nBits=self.nBits,
                                                                       voltMin=self.voltMin,
                                                                       dynamicRange=self.dynamicRange,
                                                                       offset=self.offset)

    def plotWave(self):
        self.trigGen()

        self.figure.clear()
        self.figure.tight_layout()

        ax = self.figure.add_subplot(211)
        ax.plot(self.t1, self.data1)
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Upstream (V)')
        ax.set_xlim((0, self.nsamples * self.dt))
        ax.set_ylim((self.offset, self.offset + (2 ** self.nBits)))

        ax = self.figure.add_subplot(212)
        ax.plot(self.t2, self.data2)
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Downstream (V)')
        ax.set_xlim((0, self.nsamples * self.dt))
        ax.set_ylim((self.offset, self.offset + (2 ** self.nBits)))

        self.draw()

    def findHits(self):
        self.showHitLines = not self.showHitLines

        if not self.foundHits:
            [self.hitStartIndexList1,
             self.hitLogic1,
             self.baseline1,
             self.noiseSigma1] = cfd.HitFinder(p=self.data1,
                                               noiseSigmaInVolt=self.noiseSigmaInVolt,
                                               cfdThreshold=self.cfdThreshold,
                                               nNoiseSigmaThreshold=self.nNoiseSigmaThreshold)

            [self.hitStartIndexList2,
             self.hitLogic2,
             self.baseline2,
             self.noiseSigma2] = cfd.HitFinder(p=self.data2,
                                               noiseSigmaInVolt=self.noiseSigmaInVolt,
                                               cfdThreshold=self.cfdThreshold,
                                               nNoiseSigmaThreshold=self.nNoiseSigmaThreshold)

        ax = self.figure.add_subplot(211)
        if self.showHitLines:
            for x in self.hitStartIndexList1:
                aHitLines = ax.axvline(x=x * self.dt, color='#F6553C')
                self.hitLines1.append(aHitLines)
        else:
            for aHitLine in self.hitLines1:
                aHitLine.remove()
            self.hitLines1 = list()

        ax = self.figure.add_subplot(212)
        if self.showHitLines:
            for x in self.hitStartIndexList2:
                aHitLines = ax.axvline(x=x * self.dt, color='#F6553C')
                self.hitLines2.append(aHitLines)
        else:
            for aHitLine in self.hitLines2:
                aHitLine.remove()
            self.hitLines2 = list()

        self.draw()
        self.foundHits = True

    def showTOF(self):
        self.showToFRegions = not self.showToFRegions

        if not self.foundHits:
            self.findHits()

        self.matchedHitList = tm.TimeMatching(hitListUpstream=self.hitStartIndexList1 * self.dt,
                                              hitListDownstream=self.hitStartIndexList2 * self.dt,
                                              coincidenceWindowLowerLim=10,
                                              coincidenceWindowUpperLim=50)

        ax = self.figure.add_subplot(211)
        if self.showToFRegions:
            for j in range(0, np.shape(self.matchedHitList)[0]):
                if self.matchedHitList[j, 0] == 0 and self.matchedHitList[j, 1] == 0:
                    continue
                x1 = self.matchedHitList[j, 0]
                x2 = self.matchedHitList[j, 1]
                aToFRegion = ax.fill_betweenx(y=range(self.offset, self.offset + 2 ** self.nBits),
                                              x1=x1,
                                              x2=x2,
                                              facecolor='#37A055',
                                              alpha=0.5)
                self.ToFRegion1.append(aToFRegion)
        else:
            for aToFRegion in self.ToFRegion1:
                aToFRegion.remove()
            self.ToFRegion1 = list()

        ax = self.figure.add_subplot(212)
        if self.showToFRegions:
            for j in range(0, np.shape(self.matchedHitList)[0]):
                if self.matchedHitList[j, 0] == 0 and self.matchedHitList[j, 1] == 0:
                    continue
                x1 = self.matchedHitList[j, 0]
                x2 = self.matchedHitList[j, 1]
                aToFRegion = ax.fill_betweenx(y=range(self.offset, self.offset + 2 ** self.nBits),
                                              x1=x1,
                                              x2=x2,
                                              facecolor='#37A055',
                                              alpha=0.5)
                self.ToFRegion2.append(aToFRegion)
        else:
            for aToFRegion in self.ToFRegion2:
                aToFRegion.remove()
            self.ToFRegion2 = list()

        self.draw()

    def togglePedestal(self):
        self.showPedestal = not self.showPedestal

        if not self.foundHits:
            self.showHitLines = not self.showHitLines
            self.findHits()

        ax = self.figure.add_subplot(211)
        if self.showPedestal:
            pedLine = ax.axhline(self.baseline1, color='#F36E19')
            noiseBand1 = ax.fill_between(x=np.arange(0, self.nsamples * self.dt, self.dt),
                                         y1=self.baseline1 - self.noiseSigma1,
                                         y2=self.baseline1 + self.noiseSigma1,
                                         facecolor='#F36E19',
                                         alpha=0.7)
            self.pedestalLines1.append(pedLine)
            self.pedestalLines1.append(noiseBand1)
        else:
            for lines in self.pedestalLines1:
                lines.remove()
            self.pedestalLines1 = list()

        ax = self.figure.add_subplot(212)
        if self.showPedestal:
            pedLine = ax.axhline(self.baseline2, color='#F36E19')
            noiseBand2 = ax.fill_between(x=np.arange(0, self.nsamples * self.dt, self.dt),
                                         y1=self.baseline2 - self.noiseSigma2,
                                         y2=self.baseline2 + self.noiseSigma2,
                                         facecolor='#F36E19',
                                         alpha=0.7)
            self.pedestalLines2.append(pedLine)
            self.pedestalLines2.append(noiseBand2)
        else:
            for lines in self.pedestalLines2:
                lines.remove()
            self.pedestalLines2 = list()

        self.draw()


    def toggleHitThreshold(self):
        self.showHitThreshold = not self.showHitThreshold

        if not self.foundHits:
            self.showHitLines = not self.showHitLines
            self.findHits()

        ax = self.figure.add_subplot(211)
        if self.showHitThreshold:
            hitLogicFunction = self.hitLogic1 * 100000 - 50000
            hitLogicLines, = ax.plot(self.t1, hitLogicFunction, color='#2A2A2A')
            hitThresholdLine = ax.axhline(self.baseline1 - self.nNoiseSigmaThreshold * self.noiseSigma1,
                                          color='#2A2A2A')
            self.hitThresholdLine1.append(hitThresholdLine)
            self.hitThresholdLine1.append(hitLogicLines)
        else:
            for lines in self.hitThresholdLine1:
                lines.remove()
            self.hitThresholdLine1 = list()

        ax = self.figure.add_subplot(212)
        if self.showHitThreshold:
            hitLogicFunction = self.hitLogic2 * 100000 - 50000
            hitLogicLines, = ax.plot(self.t2, hitLogicFunction, color='#2A2A2A')
            hitThresholdLine = ax.axhline(self.baseline2 - self.nNoiseSigmaThreshold * self.noiseSigma2,
                                          color='#2A2A2A')
            self.hitThresholdLine2.append(hitThresholdLine)
            self.hitThresholdLine2.append(hitLogicLines)
        else:
            for lines in self.hitThresholdLine2:
                lines.remove()
            self.hitThresholdLine2 = list()



        self.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
