"""
This module supports to figure out rssi and snr based on 

dist :distance
"""
from scipy.spatial.distance import pdist
import numpy as np
import math

class setRssi():
    def __init__(self,node1 = None,node2 = None):
       
        self.node1 = node1
        self.node2 = node2
    def getDistance(self, src, dst):
        """ Get the distance between two nodes 
       
        :param src: source node
        :param dst: destination node
        """
        pos_src = src.params['position']
        pos_dst = dst.params['position']
        points = np.array([(pos_src[0], pos_src[1], pos_src[2]), (pos_dst[0], pos_dst[1], pos_dst[2])])
        return float(pdist(points))

    def setRSSI(self):
        """set RSSI
        
        :param node1: station
        :param node2: access point
        :param wlan: wlan ID
        :param dist: distance
        """
        dist = self.getDistance(self.node1,self.node2)
        if dist < self.node1.params['range']:
            rssi = self.friisPropagationLossModel(self.node1, self.node2, dist)
        else:
            rssi = 0
        return float(rssi)  # random.uniform(value.rssi-1, value.rssi+1)       


    def setSNR(self, sta, wlan):
        """set SNR
        
        :param sta: station
        :param wlan: wlan ID
        """
        return float('%.2f' % (sta.params['rssi'][wlan] - (-90.0)))
    def pathLoss(self, node1, dist):
        """Path Loss Model:
        (f) signal frequency transmited(Hz)
        (d) is the distance between the transmitter and the receiver (m)
        (c) speed of light in vacuum (m)
        (L) System loss"""
        f = node1.params['frequency'][0] * 10 ** 9  # Convert Ghz to Hz
        c = 299792458.0
        L = 1

        if dist == 0:
            dist = 0.1
        lambda_ = c / f  # lambda: wavelength (m)
        denominator = lambda_ ** 2
        numerator = (4 * math.pi * dist) ** 2 * L
        pathLoss_ = 10 * math.log10(numerator / denominator)

        return pathLoss_
    def friisPropagationLossModel(self, node1, node2, dist):
        """Friis Propagation Loss Model:
        (f) signal frequency transmited(Hz)
        (d) is the distance between the transmitter and the receiver (m)
        (c) speed of light in vacuum (m)
        (L) System loss"""
        gr = node1.params['antennaGain'][0]
        pt = node2.params['txpower'][0]
        gt = node2.params['antennaGain'][0]
        gains = pt + gt + gr

        pathLoss = self.pathLoss(node1, dist)
        rssi = '%.2f'%(gains - pathLoss)

        return rssi

