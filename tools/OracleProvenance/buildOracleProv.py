import os
import numpy as np
import pandas as pd
import json
import argparse
from operator import itemgetter

#Parse for reference file, journal file, json list, and optional size of N
parser = argparse.ArgumentParser(description="Generate oracle provenance json files.")
parser.add_argument('-r','--inRef',type=str,default=None,\
help="Required reference file",metavar='character')
parser.add_argument('-i','--index',type=str,default=None,\
help="Required index file",metavar='character') #world idex file to make sure all files in node.csv are in world data set.
parser.add_argument('-rn','--refNode',type=str,default=None,\
help="Reference node file. A default will be picked if this option is left unselected.",metavar='character')
#parser.add_argument('-jj','--journalJoin',type=str,default=None,\
#help="required probe-journal join file",metavar='character')
#parser.add_argument('-rj','--refJournal',type=str,default=None,\
#help="required reference journal file",metavar='character')
parser.add_argument('-s','--inSys',type=str,nargs='*',\
help="Required system output files for oracle provenance construction.",metavar='character')
parser.add_argument('-oR','--outRoot',type=str,default="OracleProvJsons",\
help="Oracle provenance json output directory. Default=[OracleProvJsons]",metavar='character')
parser.add_argument('-N','--number',type=int,default=100,\
help="Number of images in each oracle provenance json file",metavar='positive integer')

args = parser.parse_args()

def printerr(string,exitcode=1):
    parser.print_help()
    print(string)
    exit(exitcode)

N = args.number

#check if reference and journal files are valid files
if args.inRef is None:
    printerr("ERROR: The reference file must be supplied.")

if args.index is None:
    printerr("ERROR: The index file must be supplied.")

if not os.path.isfile(args.inRef):
    printerr("ERROR: The reference file does not appear to be valid or existent.")

if args.inSys is None:
    printerr("ERROR: The system output file must be supplied.")
elif len(args.inSys) == 0:
    printerr("ERROR: At least one system output file must be supplied.")

if args.refNode is None:
    print("No reference node file provided. Picking a default...")
    refpfx = args.inRef.split('.')[0]
    args.refNode = refpfx + "-node.csv"
    if not os.path.isfile(args.refNode):
        printerr("ERROR: The reference node file does not appear to be valid or existent.")

#read in reference and node files
myRef = pd.read_csv(args.inRef,sep='|',header=0,na_filter=False)
index = pd.read_csv(args.index,sep='|',header=0,na_filter=False)
myNodes = pd.read_csv(args.refNode,sep='|',header=0,na_filter=False)
myNodes = pd.merge(myNodes,index,how='left',on=['WorldFileID','WorldFileName']) 
#myJournal = pd.read_csv(args.refJournal,sep='|',header=0,na_filter=False)

if not os.path.isdir(args.outRoot):
    os.system('mkdir ' + args.outRoot)
    os.system('mkdir ' + os.path.join(args.outRoot,'jsons'))

#GTProbes = myRef.query("IsTarget=='Y'")
GTProbes = myRef.copy() #NOTE: in case of further need to filter

#construct outer-joined system output by ProvenanceProbeFileID

def aggregateSystem(syslist):
    mySys=0
    dirlist = {}
    for sysfname in syslist:
        if not os.path.isfile(sysfname):
            print("{} does not appear to be a valid file. Skipping it.".format(sysfname))
            continue
    
        expid = os.path.basename(sysfname).split('.')[0]
        dirlist[expid] = os.path.dirname(os.path.abspath(sysfname))
        team = os.path.basename(sysfname).split('_')[0] #NOTE: substitute for experiment ID?
        mySysNow = pd.read_csv(sysfname,sep='|',header=0,na_filter=False)
        mySysNow.rename(index=str,columns={'ProvenanceOutputFileName':"{}ProvenanceOutputFileName".format(team),
                                           'ConfidenceScore':"{}ConfidenceScore".format(team),
                                           'IsOptOut':"IsOptOut{}".format(team)},inplace=True)
        if mySys is 0:
            mySys = mySysNow.copy()
        else:
            mySys = pd.merge(mySys,mySysNow,how='outer',on=["ProvenanceProbeFileID"])
    return mySys,dirlist

mySys,dirlist = aggregateSystem(args.inSys)

jsoncsv = []
for i,row in GTProbes.iterrows():
    nodes = []
    jsoncsv.append(pd.DataFrame({'ProvenanceProbeFileID':row['ProvenanceProbeFileID'],'ProvenanceOutputFileName':"jsons/{}.json".format(row['ProvenanceProbeFileID'])},index=[i]))
    #Get a list of GT images with the same journal (including the probe) from node reference file
    GTList = myNodes.query("ProvenanceProbeFileID==['{}']".format(row['ProvenanceProbeFileID']))

    for j,gtrow in GTList.iterrows():
        #add to nodes array as dicts with file and fileid
        nodes.append({'file':gtrow['WorldFileName'],'fileid':gtrow['WorldFileID'],'nodeConfidenceScore':1,'id':"id{}".format(j)}) #NOTE: nodeConfidenceScore is only to check

    #add the rest N - number of GT images from highest average confidence scores from the input jsons listed that are not already in the array under sys
    if len(nodes) < N:
        GTFiles = GTList['WorldFileID'].tolist()
        nodes_left = N - len(nodes)
        tempSys = 0
        for d in dirlist.keys():
            team = d.split('_')[0]
            sysjs = mySys.query("ProvenanceProbeFileID==['{}']".format(row['ProvenanceProbeFileID']))["{}ProvenanceOutputFileName".format(team)]
            if len(sysjs) == 0:
                print("ProvenanceProbeFileID {} not found in sysjs. Continuing...".format(row['ProvenanceProbeFileID'])) 
            sysjs = sysjs.iloc[0]
            with open(os.path.join(dirlist[d],sysjs)) as jsonfile:
                js = json.load(jsonfile)
                jsdfs = [pd.DataFrame(e,index=[idx]) for idx,e in enumerate(js['nodes'])]
                jsdf = pd.concat(jsdfs).drop('id',1)

                #rename nodeConfidenceScore
                scoreCol = "{}NodeScore".format(team)
                rankCol = "{}NodeRank".format(team)
                jsdf.rename(index=str,columns={"nodeConfidenceScore":scoreCol},inplace=True)
                #normalize scores from 0 to 1 linearly (i.e. divide all scores by max - min if nonzero. If max - min = 0, set all scores to 0.5.)
                jsdf[rankCol] = jsdf.rank(numeric_only=True,ascending=False)[scoreCol]

                if tempSys is 0:
                    tempSys = jsdf.copy()
                else:
                    tempSys = pd.merge(tempSys,jsdf,how='outer',on=["fileid","file"])

        #filter out all the nodes from dataframe already in nodes
        tempSys = tempSys.query("fileid != {}".format(GTFiles))
        rankCols = [c for c in list(tempSys) if "NodeRank" in c]
        for c in rankCols:
            tempSys[c].fillna(value=tempSys[c].max(),inplace=True) #set all NA or np.nan scores to max rank
        if len(rankCols) == 1:
            tempSys['average'] = tempSys[rankCols]
        else:
            tempSys['average'] = tempSys[rankCols].mean(axis=1) #average ranks across column scores
        tempSys.sort_values(by='average',inplace=True)
        tempSys.to_csv(path_or_buf=os.path.join(args.outRoot,"{}-log.csv".format(row['ProvenanceProbeFileID'])),index=False)

        #append however many nodes left into nodes, or until tempSys is empty
        myid=N
        while (nodes_left > 0) and (len(tempSys) > 0):
            temprow = tempSys.iloc[0]
            nodes.append({'file':temprow['file'],'fileid':temprow['fileid'],'nodeConfidenceScore':1,'id':'id{}'.format(myid)}) #NOTE: nodeConfidenceScore is only to check
            tempSys.drop(tempSys.index[:1],inplace=True)
            myid=myid+1
            nodes_left = nodes_left - 1 

    nodes = sorted(nodes,key=itemgetter('fileid'))
    for idx,n in enumerate(nodes):
        n['id'] = 'id{}'.format(idx)

    jsonout = {'directed':True,'nodes':nodes}
    with open(os.path.join(args.outRoot,"jsons/{}.json".format(row['ProvenanceProbeFileID'])),'w') as outfile:
        json.dump(jsonout,outfile,indent = 4)

#output valid csv to directory
jsoncsv = pd.concat(jsoncsv)
jsoncsv = jsoncsv[["ProvenanceProbeFileID","ProvenanceOutputFileName"]]
jsoncsv.to_csv(path_or_buf=os.path.join(args.outRoot,"{}.csv".format(args.outRoot.split("/")[-1])),index=False,sep="|")
