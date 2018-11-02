def traditionalControl(self):
    agentAP = None
    currentID = self.connected_tag(self.sta14)

    if currentID == 0:
            agentAP = self.ap1
    elif currentID == 1:
            agentAP = self.ap2
    elif currentID == 2:
            agentAP = self.ap3
    elif currentID == 3:
            agentAP = self.ap4
    elif currentID == 4:
            agentAP = self.ap5
    elif currentID == 5:
            agentAP = self.ap6
    elif currentID == 6:
            agentAP = self.ap7
    elif currentID == 7:
            agentAP = self.ap8
    elif currentID == 8:
            agentAP = self.ap9
    else:
            pass

    valueOfCurrentRSSI = self.chanFunt(agentAP, self.sta14)
    if valueOfCurrentRSSI <= pre_setThreshold:
        rssi = -100.0
        for each_item in range(0,len(self.observation)):
            if self.observation[each_item]!= 0:
                if self.observation[each_item] > rssi:
                    rssi = self.observation[each_item]
                else:
                    pass
            else:
                pass
    if rssi == -100:
            pass
    else:
        nextAP = self.observation.index(rssi)
        for idx, wlan in enumerate(self.sta14.params['wlan']):
            wlan = idx
        if nextAP != currentID:
            print "handoff..."
            self.handover(self.sta14, nextAP, wlan, 0)
        else:
            print "keep current state..."