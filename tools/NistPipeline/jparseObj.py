import json
import argparse
from pprint import pprint
import os, sys, subprocess
#from __future__ import print_function
import glob
#import dimensions

class Assign:
    'Class documentation: Json info'
    assigncnt = 0
    def __init__(self, f):
        self.jfile = f
        #self.html = html
        #self.imgdir=os.path.dirname(self.jfile)
        #self.id=os.path.basename(self.imgdir)
        with open(self.jfile) as data:
            #print type(data)
            self.jsondata = json.load(data)
        #self.getComp()
        #self.__findEnd()
        #if Assign.assigncnt==0:
        #    self.__createhtml()
        Assign.assigncnt += 1

    def __findImg(self, tofind):
        f=()

        for x in range(0, len(self.jsondata["nodes"])):

            try:
                if self.jsondata["nodes"][x]["nodetype"]==tofind:
#                    f.append(x)
                    
                    f=f+((self.jsondata["nodes"][x]["file"],x),)
            except KeyError:
#                f.append(0)
#                f.append("")
                continue
        return f

    def getDonorImg(self):
        return self.__findImg("donor")

    def getFinalImg(self):
        return self.__findImg("final")

    def getBaseImg(self):
        return self.__findImg("base")

    def getMeta(self, field):
        try:
            return self.jsondata["graph"][field]
        except:
            return None
            
    def getOps(self):
        ops=[]
        
        for x in range(0, len(self.jsondata["links"])):
            ops.append(self.jsondata["links"][x]["op"])
            
        self.ops=ops

    def findOp(self):
         #        print "test"
         if any("Splice" in s for s in self.ops):
             print "here"

    def findDon(self):
        print "test"

    def __createhtml(self):
        f=open(self.html,'w')
        message= """<!DOCTYPE html>
<html>
  <head>
    <title>Images</title>
  </head>
  <body>
  <table style="width:100%" border="2">
   <tr>
   <th>Project ID</th>
   <th>Base Image </th>
   <th>Manipulated Image </th>
   <th>Composite Mask </th>
   </tr>
"""
        f.write(message)
        f.close()
        
    def writehtml(self):
        f=open(self.html,'a')
        message="""
   <tr>
    <td align=\"center\">"""+self.id+"""</td>
    <td align=\"center\"><a
    href="""+self.imgdir+"""/"""+self.begfile+"""><img
    src="""+self.imgdir+"""/"""+self.begfile+"""
    height=\"100\" width=\"124\"></a>
</td>
    <td align=\"center\"><a
    href="""+self.imgdir+"""/"""+self.endfile[0]+"""><img
    src="""+self.imgdir+"""/"""+self.endfile[0]+"""
    height=\"100\" width=\"124\"></a>
</td>
    <td align=\"center\"><a
    href="""+self.composite+"""><img
    src="""+self.composite+"""
    height=\"100\" width=\"124\"></a>
</td>
</tr>
"""
        f.write(message)
        f.close()

    def endhtml(self):
        f=open(self.html,'a')
        message="""
</table>
</body>
</html>"""

        f.write(message)
        f.close()
    def __createIndex(self, ind):
        f=open(ind,'w')
        message="TaskID|ProbeFileID|ProbeFileName|ProbeWidth|ProbeHeight\n"
        f.write(message)
        f.close()

    def getImgWidthHeight(self, img):
        try:
            d=dimensions.dimensions(img)[:2]
            return d
        except:
            return None
    def getImgDir(self):
        return os.path.dirname(self.jfile)
    def getAllMeta(self):
        allmet={}
        met=["manipulationcategory","manmade","face","people","largemanmade","landscape","othersubject","manipulationpixelsize","seamcarving","prnu","imagingcompression","launderingsocialmedia","launderingmedianfiltering","remove","splice","clone","resize","warping","blurlocal","healinglocal","histogramnormalizationglobal"]
        for m in met:
            allmet[m]=self.getMeta(m)

        return allmet

    def writeIndex(self, ind, task):
        #TaskID|ProbeFileID|ProbeFileName|ProbeWidth|ProbeHeight
        if not os.path.isfile(ind):
            self.__createIndex(ind)
        finimg=self.__findImg("final")
        print finimg
        projdir=self.getImgDir()
        print projdir+str(finimg[0])
        
        dim=self.getImgWidthHeight(projdir+"/"+str(finimg[0]))
        if dim==None:
            print "Final Image was not successfully found for %s" %self.jfile
        
        
        line=task+"|"+finimg[0]+"|"+finimg[0]+"|"+str(dim[0])+"|"+str(dim[1])+"\n"
        print line
        
    def TestMaker(self, outdir, indir):
        #outdir - Base directory for index, references, images
        #indir  - Base directory for many Journal directories
        if not os.path.exists(indir):
            print "Input Directory Does Not Exist! No Exiting!"
            sys.exit()
        
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        
        ind=outdir+"/index"
        if not os.path.exists(ind):
            os.makedirs(ind)

        pro=outdir+"/probe"
        if not os.path.exists(ind):
            os.makedirs(ind)

        ref=outdir+"/reference"
        if not os.path.exists(ref):
            os.makedirs(ref)
        
        


    def writeManipRef(self, task, coll):
        #TaskID|ProbeFileID|ProbeFileName|ProbeMaskFileName|IsTarget|CompositeMaskManipulationPixelSize|Collection|JsonProjID|BaseFileName|ManipulationCategory|ManMade|Face|People|LargeManMade|Landscape|OtherSubject|Remove|Splice|Clone|Resize|Warping|BlurLocal|HealingLocal|HistogramNormalizationGlobal|Contrast|Sharpen|Color|Cropping|SeamCarving|OtherEnhancements|CGI|Watermark|Steganography|Reformatting|ImagingCompression|Recapture|Mosaicking|Restaging|Repurposing|LaunderingSocialMedia|LaunderingMedianFiltering|AntiForensicIllumination|AntiForensicAddCamFingerprint|AntiForensicsCompressionTable|AntiForensicNoiseRestoration|AntiForensicAberrationCorrection|AntiForensicCFACorrection|AntiForensicOther|OriginalImgWithoutAntiForensic|PRNU|PRNUFileSetID|RemoveMaskManipulationPixelSize|SpliceMaskManipulationPixelSize|CloneMaskManipulationPixelSize|DonorFileSetID|DonorFileMaskFileSetID
        #Manipulation|NC2016_0128|probe/NC2016_0128.jpg|reference/manipulation/mask/NC2016_3942.png||||Y|none||simple|N|Y|N|Nimble-SCI|world/NC2016_3272.jpg|same|N|probe/NC2016_7105.jpg|
        #TaskID|ProbeFileID|ProbeFileID
#        print "manipulation"

        meta=self.getAllMeta()
        print task    #TaskID
        print coll
        print self.__findImg("final")   #probeFile
        
        print meta
        

    def getComp(self):
        comp=glob.glob(self.imgdir+'/*composite_mask*')
        print comp
        try:
           self.composite=comp[0]
        except:
            self.composite=""


#j="11336819694_2364a6a435_o.json"
#a=Assign(j)
#x=a.getMeta("manipulationcategory")
#print x
#x=a.getFinalImg()
#print x

#path="/Volumes/medifor/JournalingTool/JTSite"
#htmlfile=path+"/"+"index2.html"
#dirs=os.listdir(path)
#cnt=0
#for date in dirs:
##    if date.startswith("."):
#    if "." in date:
#        continue
#    curpath=path+"/"+date
#    dirsin=os.listdir(curpath)
#    for assign in dirsin:
##        if assign.endswith(".tgz") or assign.startswith("."):
#        if "." in assign:
#            continue
#        print curpath+assign
#        findj=os.path.basename(subprocess.check_output("find "+curpath+"/"+assign+"/ -name '*.json' ! -name '.*'",shell=True)).splitlines()
#        jsonfile=curpath+"/"+assign+"/"+findj[0]
#        print jsonfile
##        sys.exit()
#        curAssign=Assign(jsonfile,htmlfile)
##        curAssign=Assign('/Volumes/medifor/JournalingTool/Delivery0907/Andrew/165d6da20141854378c9ce7ac7129ea0/165d6da20141854378c9ce7ac7129ea0.json','/Volumes/medifor/JournalingTool/Delivery0907/Andrew/165d6da20141854378c9ce7ac7129ea0.html')
##        sys.exit()
#        print curAssign.file
#        print curAssign.html
#        print curAssign.imgdir
#        print curAssign.id
#
#        curAssign.writehtml()
#        #cnt=cnt+1

        
#curAssign.endhtml()
##curAssign.getOps()
##curAssign.findSplice()
##print curAssign.ops
##print curAssign.endfile
##print curAssign.begfile
##print curAssign.donfile
##end=[]
##with open('/Users/apd2/Downloads/ef2a5ab2a5220404ac1d285d6eae09ac/ef2a5ab2a5220404ac1d285d6eae09ac.json') as data:
##    pprint(data)
##    p_json = json.load(data)
##    #    node = locateByName(p_json,'0')
##    print len(p_json['nodes'])
##    source= [item["source"] for item in p_json["links"]

