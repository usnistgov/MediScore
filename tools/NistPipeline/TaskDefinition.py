import json
import argparse
from pprint import pprint
import os, sys, subprocess
#from __future__ import print_function
import glob
import dimensions

class Task:
    def __init__(self, journalList, hpList):
    #    self.jlist=journalList
    #    self.hpList
        

    def isTarg(self):
        print "test"

    def getProbe(self):
        print "test"
        f=[]

        for x in range(0, len(self.jsondata["nodes"])):

            try:
                if self.jsondata["nodes"][x]["nodetype"]=="final":
#                    f.append(x)
                    f.append(self.jsondata["nodes"][x]["file"])
            except KeyError:
                f.append("")

        return f

    
#    def getMD(self):
#        print "test"

#    def getMask(self):
#        print "test"

#    def getMaskDonor(self):
#        print "test"

#    def getDonor(self):
#        print "test"
