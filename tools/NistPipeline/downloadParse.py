import subprocess
#from lxml import etree
#import lxml.html as LH
from bs4 import BeautifulSoup
import re
import os, sys
import datetime
import urllib
import tarfile

cmd="wget --load-cookies cookies.txt -O out.html http://medifor.rankone.io/journal/ "
process=subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

rfile=open('out.html')
rsoup=BeautifulSoup(rfile, "lxml")
#nodes1 =rsoup.find('div',{'class':'rows'})
headers=rsoup.find("table", {"id" : "journals"}).findAll("th")
#print headers
values=rsoup.find("table", {"id" : "journals"}).findAll("td")
#print values
now=datetime.datetime.now()
h=[]
path="/Volumes/medifor/JournalingTool/JTSite"
cnt=0
mcnt=1
full={}
for line in headers:
    hh=re.findall(r'>(.*?)<',str(line))
    h.append(hh[0])


full[mcnt]={}
for line in values:
    dchk=0
    if cnt==8:
        cnt=0
        mcnt=mcnt+1
        full[mcnt]={}
    vv=re.findall(r'>(.*?)<',str(line))
    #print vv
    if cnt==1:
        full[mcnt][h[cnt]]=vv[1]
        dirs=os.listdir(path)
        for file in dirs:
            testpath=path+"/"+file+"/"+str(vv[1])
            if os.path.isdir(testpath)==True:
                dchk=1
               # print "here"
                break
#        if dchk!=1:
            
            
    else:
        full[mcnt][h[cnt]]=vv[0]
    cnt=cnt+1

curdir=str(now.strftime("%m")) + str(now.strftime("%d"))

bdownpath=path+"/"+curdir
try:
    os.stat(bdownpath)
except:
    os.mkdir(bdownpath)

#Parse html of project page to get download link
basecmd="https://s3.amazonaws.com/medifor/browser/projects/"
todownl=[]
for k in full.keys():
    dchk=0
#    print full[k]['ID']
    for f in dirs:
        testpath=path+"/"+f+"/"+full[k]['Name']
        if os.path.isdir(testpath)==True:
            dchk=1
            
            # print "here"
            break
    if dchk!=1:
        todownl.append(k)

total=len(todownl)
it=1
print "Downloading %d projects" %total
for k in todownl:
    print "Currently downloading %d of %d" %(it, total)
    it=it+1
    downpath=bdownpath+"/"
    getfile=basecmd+full[k]['Name']+".tgz"        
    #print cmd
    #        testfile=urllib.URLopener()
    #        testfile.retrieve(cmd,downpath)
    print "downloading " + full[k]['Name'] + ".tgz"
    downcmd="wget -P "+downpath+" "+getfile
    #        print downcmd
    #        sys.exit()
    process=subprocess.Popen(downcmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print "Download finished. Now extracting"
    
    try:
        tar=tarfile.open(downpath+full[k]['Name']+".tgz","r")
        tar.extractall(path=bdownpath+"/")
        tar.close()
    except:
        print "Can not decompress " +full[k]['Name']+".tgz"
        continue
    print "Finished decompressing"
    
        #print h[0]

#print full
#print h
#print vv




