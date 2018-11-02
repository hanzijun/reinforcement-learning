#!/usr/bin/python

"""
This is the module used to realize reinforcement learning
environment: WLAN env
agent: sta14
reward: throughput
state: rssi values or snr values
"""
import re
import random
import time
import math
import numpy as np
import threading
import matplotlib.pylab as plt
from scipy.spatial.distance import pdist
#from associationcontrol import associationControl
from mininet.net import Mininet
from mininet.node import Controller, OVSKernelAP
from mininet.link import TCLink
from mininet.cli import CLI
from Brain import BrainDQN
from getrssi import setRssi
from RL_brain import DeepQNetwork
from RNNBrain import RNNNetwork
from mininet.wifiMobility import mobility
from mininet.wifiLink import Association
from mininet.wifiAssociationControl import associationControl
from mininet.log import  error, debug, setLogLevel
from mininet.util import   waitListening

class mininet():
    def __init__(self):
        self.observation=None
        self.state=None
        self.reward=None
        self.valid = False
        self.AP_=None
        self.id4ap=None
        self.aplist=[]
        self.timeinterval = 0.01
        self.seqlen=30
        self.second = self.sleeptime(0, 0, 0.5)
        self.threads=[]
        self.setlog()
        self.topology()
        pass

    def chanFunt(self, new_ap, new_st):

        """collect rssi from aps to station
           :param new_ap: access point
           :param new_st: station
        """
        APS = ['ap1', 'ap2', 'ap3', 'ap4', 'ap5', 'ap6', 'ap7', 'ap8', 'ap9']
        for number in APS:
            if number == str(new_ap):
                indent = 0
                for item in new_st.params['apsInRange']:
                    if number in str(item):
                        for each_item in new_ap.params['stationsInRange']:
                            if str(new_st) in str(each_item):
                                return new_ap.params['stationsInRange'][each_item]
                                # a.append(ap1.params['stationsInRange'].values()[level+1])
                            else:
                                pass
                    else:
                        indent = indent + 1
                if indent == len(new_st.params['apsInRange']):
                    return 0
            else:
                pass

    def setSNR(self, signal):
        """
        set SNR
        :param signal: RSSI
        """
        if signal != 0:
            snr = float('%.2f' % (signal - (-91.0)))
        else:
            snr = 0
        return snr


    def connected_tag(self, new_string):
        """output the number of associated AP"""
        for each_item in new_string.params['associatedTo']:
            if str('ap1') in str(each_item):
                return 0
            elif str('ap2') in str(each_item):
                return 1
            elif str('ap3') in str(each_item):
                return 2
            elif str('ap4') in str(each_item):
                return 3
            else:
                return 0

    def sleeptime(self, hour, minu, sec):

        """time to output"""
        return hour * 3600 + minu * 60 + sec

    def step(self, current, action):
        """
        The function to realize environment 
        changes after agent doing actions 
        """
        if action == -1 :
            reward = 0

        elif action != current and self.observation[action] != 0:
            print "handoff..."
            if action == 0:
                self.AP_ = self.ap1
            elif action == 1:
                self.AP_ = self.ap2
            else :
                self.AP_ = self.ap3

            for idx, wlan in enumerate(self.sta14.params['wlan']):
                wlan = idx

            self.handover(self.sta14,self.AP_,wlan)
            #time.sleep(self.second)

            if self.observation[action]<-46:
                reward = 0.1
                time.sleep(0.5)
            else:
                reward = self._getreward()
        else: 
            print "keep current state..."
            if self.observation[action]<-46:
                reward = 0.1
                time.sleep(0.5)
            else:
                reward = self._getreward()
        nextstate = np.array(self.observation)
        return reward, nextstate

    def handover(self, sta, ap, wlan):
        '''
        the function to realize transfer association
        :param sta: station
        :param ap: target_AP
        :param wlan:  OVSAP
        :return:  new connection
        '''
        if ap not in sta.params['associatedTo']:
            cls = Association
            #debug('iwconfig %s essid %s ap %s\n' % (sta.params['wlan'][wlan], ap.params['ssid'][0], \
            #                                        ap.params['mac'][0]))
            #sta.pexec('iwconfig %s essid %s ap %s' % (sta.params['wlan'][wlan], ap.params['ssid'][0], \
            #                                          ap.params['mac'][0]))
            cls.associate_noEncrypt(sta, ap, wlan)
            mobility.updateAssociation(sta, ap, wlan)

    def _parseIperf(self, iperfOutput):

        r = r'([\d\.]+ \w+/sec)'
        #r = r'([\d\.]+\w)'
        m = re.findall(r, iperfOutput)

        if m:
            return m[-1]
        else:
            # was: raise Exception(...)
            error('could not parse iperf output: ' + iperfOutput)
            return ''

    def iperf(self, hosts=None, l4Type='TCP', udpBw='10M',
              seconds=5, port=5001):
        #t_begin = time.time()
        hosts = hosts or [hosts[0], hosts[-1]]
        assert len(hosts) == 2
        client, server = hosts
        conn1 = 0
        conn2 = 0
        if client.type == 'station' or server.type == 'station':
            if client.type == 'station':
                while conn1 == 0:
                    conn1 = int(client.cmd('iwconfig %s-wlan0 | grep -ic \'Link Quality\'' % client))
            if server.type == 'station':
                while conn2 == 0:
                    conn2 = int(server.cmd('iwconfig %s-wlan0 | grep -ic \'Link Quality\'' % server))
        
        server.cmd('killall -9 iperf')
        iperfArgs = 'iperf -p %d ' % port
        bwArgs = ''
        if l4Type == 'UDP':
            iperfArgs += '-u '
            bwArgs = '-b ' + udpBw + ' '
        server.sendCmd(iperfArgs + '-s')
        if l4Type == 'TCP':
            if not waitListening(client, server.IP(), port):
                raise Exception('Could not connect to iperf on port %d'
                                % port)
        cliout = client.cmd(iperfArgs + '-t %d -c ' % seconds +
                            server.IP() + ' ' + bwArgs)
        debug('Client output: %s\n' % cliout)
        servout = ''
        count = 2 if l4Type == 'TCP' else 1
        while len(re.findall('/sec', servout)) < count:
            servout += server.monitor(timeoutms=5000)
        server.sendInt()
        servout += server.waitOutput()
        debug('Server output: %s\n' % servout) 
        result = [self._parseIperf(servout), self._parseIperf(cliout)]
        if 'Mbit' in result[0]:
            result_new = re.findall(r'([\d\.]+\w)',result[0])[0]
        else:
            result_new = float(re.findall(r'([\d\.]+\w)',result[0])[0])/1024.0
           
        return result_new

    def _getspace(self):

        return len(self.observation), len(self.observation)

    def _getreward(self):
        if sum(self.observation)!=0:     
            reward = self.iperf([self.sta14, self.h1],l4Type='TCP',\
                                 seconds=0.00001, port=5001)
        else:
            reward = 0 
        return reward

    def _getstate(self):
        """
        Get RSSI values by the timeinterval, fresh the 
        envieonment and read the state from WLAN
        """
        try:
            while True:
                try:
                    a1,b1,c1 = setRssi(self.ap1,self.sta14),\
                               setRssi(self.ap2,self.sta14),\
                               setRssi(self.ap3,self.sta14)

                    rssi_dic = [a1.setRSSI(),
                               b1.setRSSI(),
                               c1.setRSSI()]
                except KeyboardInterrupt:
                    pass
                else:
                    if self.obsevation is None:
                        self.obsevation = np.array([rssi_dic])
                    elif self.obsevation.shape[0] == self.seqlen:
                        obsevation = np.delete(self.obsevation, (0), axis=0)
                        obsevation = np.append(obsevation, [rssi_dic], axis=0)
                        self.obsevation = obsevation
                        if not self.valid:
                            self.valid = True
                    else:
                        self.obsevation = np.append(self.obsevation, [rssi_dic], axis=0)
                finally:
                    self.currentID = self.connected_tag(self.sta14)
                    time.sleep(self.timeinterval)
        except:
            pass

    def _getstatestart(self):
        t1 = threading.Thread(target=self._getstate)
        self.threads.append(t1)
        for t in self.threads:
            t.setDaemon(True)
            t.start()
        print '**********start getting wlan states***********'

    def setlog(self):
        setLogLevel('info')

    def topology(self):
            "Create a network."
            net = Mininet(controller=Controller, link=TCLink, accessPoint=OVSKernelAP)

            print "*** Creating nodes"
            self.sta14 = net.addStation('sta14', mac='00:00:00:00:00:15', ip='10.0.0.15/8', range=10)
            self.h1 = net.addHost('h1', mac='00:00:00:00:00:01', ip='10.0.0.1/8')
            self.ap1 = net.addAccessPoint('ap1', ssid='ssid-ap1', mode='g', channel='1', position='10,20,0', range=35)
            self.ap2 = net.addAccessPoint('ap2', ssid='ssid-ap2', mode='g', channel='1', position='44,20,0', range=35)
            self.ap3 = net.addAccessPoint('ap3', ssid='ssid-ap3', mode='g', channel='1', position='27,50,0', range=35)
            #self.ap4 = net.addAccessPoint('ap4', ssid='ssid-ap4', mode='g', channel='1', position='55,55,0', range=35)
            c1 = net.addController('c1', controller=Controller)

            print "*** Configuring wifi nodes"
            net.configureWifiNodes()

            print "*** Associating and Creating links"
            net.addLink(self.h1,self.ap1)
            net.addLink(self.ap1,self.ap2)
            net.addLink(self.ap2,self.ap3)
            #net.addLink(self.ap3,self.sta14)

            """uncomment to plot graph"""
            net.plotGraph(max_x=55, max_y=55)

            """association control"""
            #net.associationControl("ssf")

            """Seed"""
            net.seed(13)
            """random walking"""
            net.startMobility(time=0, model='RandomDirection', max_x=55, max_y=55, min_v=0.09,  max_v=0.09)

            print "*** Starting network"
            net.build()
            c1.start()
            self.ap1.start([c1])
            self.ap2.start([c1])
            self.ap3.start([c1])
            #self.ap4.start([c1])
            #net.startMobility(startTime=0)
            #net.mobility(self.sta14, 'start', time=0, position='20,30,0')
            #net.mobility(self.sta14, 'stop', time=101, position='19.9,29.9,0')
            #net.stopMobility(stopTime=3600)


            print "*** Running CLI"
            CLI(net)
            self._getstatestart()
            while not self.valid:
                time.sleep(self.second)

            n_actions, n_features = self._getspace()
            brain = BrainDQN(n_actions,n_features,60, param_file = None)
            state = np.array(self.observation)
            print 'initial observation:' + str(state)

            #self.aplist=['ap1','ap2','ap3','ap4']
            #self.id4ap = dict(zip(xrange(0,n_actions),self.aplist))
            #print self.id4ap
            #data = {}
            #fig = Display(self.id4ap)
            #fig.display()
            try:
                while True:

                    action,q_value = brain.getAction(state)
                    print "action:  "+str(action)+"  Q:  "+str(q_value)
                    reward, nextstate = self.step(self.currentID, action)
                    print "reward:  "+str(reward)+'  state_:  '+str(nextstate)

                    #data['timestamp'] = time.time()
                    #data['rssi'] = nextstate
                    #data['q'] = q_value
                    #data['reward'] = reward
                    #data['action_index'] = action
                    #print 'DATA:  ',data
                    #fig.append(data)

                    brain.setPerception(state, action, reward, nextstate, False)
                    state = nextstate

            except KeyboardInterrupt:
                print 'saving replayMemory...'
                brain.saveReplayMemory()
                #fig.stop()
            pass
            # print new_rssi
            # snr_dict = map(setSNR,new_rssi)
            print "*** Stopping network"
            net.stop()

if __name__=='__main__':
    mininet()

