import sys, os
sys.path.insert(0,'/Users/apd2/Downloads/maskgen-master11/maskgen')  #'/Users/apd2/Downloads/maskgen-master11/maskgen') /home/medifor/medifor/JournalingTool/JTool/maskgen-master20161107/maskgen

from scenario_model import ImageProjectModel
from abc import ABCMeta, abstractmethod
from TaskData import TaskDefinition


class TaskDefManipulation(TaskDefinition):
    def __init__(self, task, dataobj):
        self.task=task
        self.data=dataobj
    def test(self):
        print self.task

    def isTarg(self, tup):
        
        if tup[0]=="hp" or tup[0]=="world":
            return "N"
        elif tup[0]=="jt":
            if self.task=="manipulation":
                return "Y"
    def getTrials(self):
        trials=()
        self.getProbes()
        return self.probes

    def getMask(self, tup):
        if tup[0]=="hp":
            maskname=''
        elif tup[0]=="jt":
            name, ext=os.path.splitext(tup[1])
            maskname=str(self.data.jtjson.files[tup[1]]['loc'])+"/"+str(name)+"-compMask.png"
            if not os.path.isfile(maskname):
                print name
                print self.data.jtjson.files[tup[1]]['json']
                print maskname
                model=ImageProjectModel(self.data.jtjson.files[tup[1]]['json'])
                model.selectImage(name)
                model.getComposite().toPIL().save(maskname)
        return ('mask', maskname,)
        

class TaskDefSplice(TaskDefinition):
    def __init__(self, task, dataobj):
        #super(TaskDefManipulation, self).__init__(task, dataobj)
        self.task=task
        self.data=dataobj
    
    def getTrials(self):
        trials=()
        self.getProbes()
        self.getDonors()
#        print self.donors
        print len(self.probes)
        print len(self.donors)
        for p in self.probes:
            self.getDonorsForProbe(p)
            for d in self.donors:
                trials=trials+((p,d),)
        return trials

    def isTarg(self,tup):
        """tup must include (type "hp", probe, donor, dataObj)"""
        probetup=tup[0]
        donortup=tup[1]
        if donortup[0]=="world":
            return "N"
        if probetup[0]=="hp":
            return "N"
        elif probetup[0]=="jt":
            if any(donortup[1] in f for f in self.data.jtjson.files[probetup[1]]['donors']):
                return "Y"
            else:
                return "N"

#    def getDonorFiles(self, dataObj):
        
    def getDonorsForProbe(self, tup):
        self.data.getDonorsForImage(tup)

    def getDonorMask(self, tup):
#        src=tup[1]
        count=0
        #    pprint(data)
        dmaskname=self.data.jtjson.donfiles[tup[1][1]]['loc']+"/"+os.path.splitext(tup[1][1])[0]+"-donmask.png"

        if os.path.isfile(dmaskname):
            return ('mask',dmaskname)
            
        jdata = self.data.jtjson.jsonfiles[self.data.jtjson.donfiles[tup[1][1]]['json']]['data']
        for x in range(0,len(jdata['nodes'])):
            if jdata['nodes'][x]['file']==tup[1][1]:
                src=x
                break

        while (count < 8):
            targ= [[item["target"],item["op"]] for item in jdata["links"]
                   if item["source"] == src]
            print targ
            if targ[0][1]=="Donor":
                othop=[item["op"] for item in jdata["links"]
                       if item["source"]!=src and item["target"]==targ[0][0]]
                if othop[0]=="PasteSplice":
                    #create donor masks
                    model=ImageProjectModel(self.data.jtjson.donfiles[tup[1][1]]['json'])
                    model.selectImage(jdata["nodes"][targ[0][0]]["id"])
                    model.getDonor().toPIL().save(dmaskname)
                    print "Found Donor"
                    print jdata["nodes"][targ[0][0]]["id"]
                    break
            else:
                src=targ[0][0]
                print src
            count=count+1
        return ('mask',dmaskname)
