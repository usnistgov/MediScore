import json
import argparse
from pprint import pprint
import os, sys, subprocess
#from __future__ import print_function
import glob
import dimensions
from jparseObj import Assign
import hashlib
#from PIL import Image
from abc import ABCMeta, abstractmethod
from PIL import Image

class WHPData:
    def __init__(self, j, typ, loc):
        self.typ=typ
        self.jfile=j
        with open(self.jfile) as data:
            #print type(data)
            self.jsondata = json.load(data)
        self.files=self.__getFileInfo(loc)

    def __getFileInfo(self, loc):
        f={}
        for x in range(0, len(self.jsondata["data"])):
            try:
                n=self.jsondata["data"][x]["file_name"].split('/')[-1]
                f[n]={}
                img=loc+"/"+n
                f[n]['md5']=hashlib.md5(open(img, 'rb').read()).hexdigest()
                if self.typ=='hp':
                    f[n]['height']=self.jsondata["data"][x]["data"]["height"]
                    f[n]['width']=self.jsondata["data"][x]["data"]["width"]
                elif self.typ=='world':
                    f[n]['height']=self.jsondata["data"][x]["height"]
                    f[n]['width']=self.jsondata["data"][x]["width"]
            except KeyError:
                
                print "FILEINFO ERROR"

        return f

class TaskDefinition:

    __metaclass__ = ABCMeta

    def __init__(self, task, dataobj):
        #self.td=Data(journalList, hpj, hpdir, testdir)
        self.task=task
        self.data=dataobj

    def getDonors(self):
        donors=()
        for i in self.data.jtjson.donfiles.keys():
            donors=donors+(('jt',i,self.data.jtjson.donfiles[i]['loc']),)
        for i in self.data.wjson.files.keys():
            donors=donors+(('world',i,),)

        self.donors=donors
        return donors
    def getProbes(self):
        #returns a tuple (type, id) type="hp, jt, world" id="manipulation,removal"
        #print len(self.td.hpjson.files)
        probes=()
        for i in self.data.hpjson.files.keys():
            probes=probes+(("hp",i,),)
        
        for i in self.data.jtjson.files.keys():
            probes=probes+(("jt",i,self.data.jtjson.files[i]['loc']),)
        self.probes=probes
        return probes

    def linkProbes(self, probe):
        for i in range(0,len(probe)):
            if probe[i][0]=="hp":
                src=str(self.data.hpdir)+"/"+str(probe[i][1])
                findst=self.data.testdir+"/"+str(probe[i][2])
                if not os.path.islink(findst):
                    os.symlink(src,findst)
            elif probe[i][0]=="jt":
                src=str(self.data.jtjson.files[probe[i][1]]['loc'])+"/"+probe[i][1]
                findst=self.data.testdir+"/"+str(probe[i][2])
                print src 
                print findst
                if not os.path.islink(findst):
                    os.symlink(src,findst)

    def linkMasks(self,masktup):
        for i in range(0,len(masktup)):
            if masktup[i][0]=="hp":
                continue
            elif masktup[i][0]=="jt":
                src=str(masktup[i][1])
                findst=self.data.testdir+"/"+str(masktup[i][2])
                print src
                print findst
                if not os.path.islink(findst):
                    os.symlink(src,findst)

    def linkDonors(self,tup):
        for don in tup:
            print don
            if don[0]=="jt":
                src=str(self.data.jtjson.donfiles[don[1]]['loc'])+"/"+don[1]
            elif don[0]=="world":
                src=self.data.wdir+"/"+don[1]
            findst=self.data.testdir+"/"+str(don[2])
            if not os.path.islink(findst):
                print src
                print findst
                os.symlink(src,findst)

    @abstractmethod
    def isTarg(self, tup):
        print "Should not see this"
#        if tup[0]=="hp" or tup[0]=="world":
#            return "N"
#        elif tup[0]=="jt":
#            if task=="manipulation":
#                return "Y"

class Data:
    def __init__(self, journalList, hpj, hpdir, testdir, worldj, worlddir): #, worldj):
        self.jtjson=JTData(journalList)
        self.hpjson=WHPData(hpj,'hp', hpdir)
        self.wjson=WHPData(worldj,'world', worlddir)
        self.hpdir=hpdir
        self.testdir=testdir
        self.wdir=worlddir
#        self.wjson=WHPData(worldj)
        
    def getMD5(self, tup):
        if tup[0]=="hp":
            md5=self.hpjson.files[tup[1]]['md5']  #hpdir+"/"+tup[1]
        elif tup[0]=="jt":
            try:
                md5=self.jtjson.files[tup[1]]['md5']
            except:
                md5=self.jtjson.donfiles[tup[1]]['md5']
        elif tup[0]=="mask":
            img=tup[1]
            md5=hashlib.md5(open(img, 'rb').read()).hexdigest()
        elif tup[0]=="world":
            md5=self.wjson.files[tup[1]]['md5'] #self.wdir+"/"+tup[1]
        return md5
#        return hashlib.md5(open(img, 'rb').read()).hexdigest()

    def getDim(self, tup):
        if tup[0]=="hp":
            return [self.hpjson.files[tup[1]]['width'], self.hpjson.files[tup[1]]['height']]
        elif tup[0]=="jt":
            try:
                return [self.jtjson.files[tup[1]]['width'], self.jtjson.files[tup[1]]['height']]
            except:
                return [self.jtjson.donfiles[tup[1]]['width'], self.jtjson.donfiles[tup[1]]['height']]
        elif tup[0]=="world":
            return [self.wjson.files[tup[1]]['width'], self.wjson.files[tup[1]]['height']]

    def linkProbes(self, probe):
        for i in range(0,len(probe)):
            if probe[i][0]=="hp":
                src=str(self.hpdir)+"/"+str(probe[i][1])
                findst=self.testdir+"/"+str(probe[i][2])
#                print src
#                print findst
                if not os.path.islink(findst):
                    os.symlink(src,findst)
            elif probe[i][0]=="jt":
                src=str(self.jtjson.files[probe[i][1]]['loc'])+"/"+probe[i][1]
                findst=self.testdir+"/"+str(probe[i][2])
                if not os.path.islink(findst):
                    os.symlink(src,findst)

    def getName(self,tup,md5,loc):
        #f=os.path.basename(tup[1])
        name, ext=os.path.splitext(tup[1])
        return loc+"/"+str(md5)+ext
    
    def linkMasks(self,masktup):
        for i in range(0,len(masktup)):
            if masktup[i][0]=="hp":
                continue
            elif masktup[i][0]=="jt":
                src=str(masktup[i][1])
                findst=self.testdir+"/"+str(masktup[i][2])
                print src
                print findst
                if not os.path.islink(findst):
                    os.symlink(src,findst)

    def createDirs(self):
        try:
            os.stat(self.testdir)
        except:
            os.mkdir(self.testdir)

        try:
            os.stat(self.testdir+"/probe")
        except:
            os.mkdir(self.testdir+"/probe")
        
        try:
            os.stat(self.testdir+"/world")
        except:
            os.mkdir(self.testdir+"/world")
        
        try:
            os.stat(self.testdir+"/indexes")
        except:
            os.mkdir(self.testdir+"/indexes")
       
        try:
            os.stat(self.testdir+"/reference/manipulation/mask")
        except:
            os.makedirs(self.testdir+"/reference/manipulation/mask")

        try:
            os.stat(self.testdir+"/reference/splice/mask")
        except:
            os.makedirs(self.testdir+"/reference/splice/mask")

    def getDonorsForImage(self, tup):
        sec=[]
        dons=[]
        print tup
        if tup[0]=="hp":
            """no donor"""
        elif tup[0]=="jt":
            jdata=self.jtjson.jsonfiles[self.jtjson.files[tup[1]]['json']]['data']

            for n in range(0,len(jdata['nodes'])):
                if jdata['nodes'][n]['file']==tup[1]:
                    targ=n
                    break
            cnt=0
            while cnt<20:
                src=[[item["source"],item["op"],jdata['nodes'][item["source"]]['nodetype']] for item in jdata["links"]
                       if item["target"] == targ ]#and jdata['nodes'][item["source"]]['nodetype']!='base']
#                print "Source:"
#                print src
                if len(src)==2:
                    ot=[1,0]
                    for s in range(0,len(src)):
                        if src[s][1]=="Donor":
#                            if src[s][2]=='donor' and not jdata['nodes'][src[s][0]]['file'] in dons:
#                                dons.append(jdata['nodes'][src[s][0]]['file'])
                            if src[ot[s]][1]!="PasteSplice":
                                targ=src[ot[s]][0]
                            elif src[ot[s]][1]=="PasteSplice" and src[s][2]=='donor': 
                                if not jdata['nodes'][src[s][0]]['file'] in dons:
                                    dons.append(jdata['nodes'][src[s][0]]['file'])
                                targ=src[ot[s]][0]
                            else:
                                targ=src[0][0]
                                sec.append(src[1][0])
                    
                elif len(src)==1:
                    if src[0][2]=='donor':
                        if not jdata['nodes'][src[0][0]]['file'] in dons:
                            dons.append(jdata['nodes'][src[0][0]]['file'])
                        if not sec:
                            break
                        targ=sec[-1]
                        sec.pop()
                    elif src[0][2]=='base':
                        if not sec:
                            break
                        else:
                            targ=sec[-1]
                            sec.pop()
                    else:       
                        targ=src[0][0]
                else:
                    if not sec:
                        break
                    else:
                        targ=sec[-1]
                        sec.pop()
                
                        
#                print "donors:"
#                print dons
                cnt=cnt+1
            self.jtjson.files[tup[1]]['donors']=dons
#            print dons
        #return dons
        
class JTData:
    def __init__(self, j):
#        with open()
        self.jlist=j
        self.JournalIterJson()
        self.JournalIterProbe()
        self.JournalIterDonor()
#        with open(self.jfile) as data:
#            #print type(data)
#            self.jsondata = json.load(data)
#        self.files=self.__getFileNames()
#        self.jl=len(self.jsondata["data"])

    def JournalIterJson(self):
        self.jsonfiles={}
        f=open(self.jlist, "r")
        lines=f.readlines()
        f.close
        lines=[l.strip() for l in lines]
        for l in lines:
#            print l
            with open(l) as data:
                self.jsonfiles[l]={}
                self.jsonfiles[l]['data']=json.load(data)

    def JournalIterProbe(self):
        self.files={}
        f=open(self.jlist,"r")
        lines=f.readlines()
        f.close()
        lines= [l.strip() for l in lines]
        for l in lines:
            jf=Assign(l).getFinalImg()
            for jft in jf:
                self.files[jft[0]]={}
                self.files[jft[0]]['json']=l
                self.files[jft[0]]['loc']=os.path.dirname(l)
                path=str(self.files[jft[0]]['loc'])+"/"+str(jft[0])
                dim=dimensions.dimensions(path)
                self.files[jft[0]]['width']=dim[0]
                self.files[jft[0]]['height']=dim[1]
                img=self.files[jft[0]]['loc']+"/"+jft[0]
                self.files[jft[0]]['md5']=hashlib.md5(open(img, 'rb').read()).hexdigest()
            
    def JournalIterDonor(self):
        self.donfiles={}
        f=open(self.jlist,"r")
        lines=f.readlines()
        f.close()
        lines=[l.strip() for l in lines]
        for l in lines:
            jf=Assign(l).getDonorImg()
            for jft in jf:
                self.donfiles[jft[0]]={}
                self.donfiles[jft[0]]['json']=l
                self.donfiles[jft[0]]['loc']=os.path.dirname(l)
                self.donfiles[jft[0]]['nodenumb']=jft[1]
                path=str(self.donfiles[jft[0]]['loc'])+"/"+str(jft[0])
                dim=Image.open(path).size
                self.donfiles[jft[0]]['width']=dim[0]
                self.donfiles[jft[0]]['height']=dim[1]
                img=self.donfiles[jft[0]]['loc']+"/"+jft[0]
                self.donfiles[jft[0]]['md5']=hashlib.md5(open(img, 'rb').read()).hexdigest()

        

