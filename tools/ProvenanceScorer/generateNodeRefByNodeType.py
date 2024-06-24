# Python versions >= 2.7, requires `requests` library.

"""
The module is to generate different ref-node files for different type of nodes.
Author: Xu Zhang
Email: xu.zhang@columbia.edu

Examples:
    
    Train the index and do the search    

    >>> python generateNodeRefByNodeType.py --dataset_name=NC2017_Dev1_Beta4 

"""


from __future__ import print_function
from urllib import urlretrieve
import os, sys, errno, json, requests, argparse
import csv
import pandas as pd
import shutil
from tqdm import tqdm

def err_quit(msg, exit_status=1):
    print(msg)
    exit(exit_status)

def load_csv(csv_fn, sep="|"):
    try:
        return pd.read_csv(csv_fn, sep)
    except IOError as ioerr:
        err_quit("{}. Aborting!".format(ioerr))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Node Ref File')
    parser.add_argument('--root_dir', type=str, default='/home/xuzhang/project/Medifor/data/', help='Directory to the dataset')
    parser.add_argument('--dataset_name', type=str, default='NC2017_Dev1_Beta4', help='Name of the dataset')
    parser.add_argument('--dataset_index', type=str, default='NC2017_Dev1', help='Index of the dataset')
    args = parser.parse_args()
    
    root_dir =  args.root_dir
    dataset_name =  args.dataset_name
    dataset_index =  args.dataset_index

    #read previous reference files
    probe_index_file = load_csv(open('{}{}/reference/provenancefiltering/{}-provenancefiltering-ref.csv'.format(root_dir, dataset_name, dataset_index), 'rb'), '|')
    node_file = load_csv(open('{}{}/reference/provenancefiltering/{}-provenancefiltering-ref-node.csv'.format(root_dir, dataset_name, dataset_index), 'rb'), '|')
    node_dict = {}
    
    #build a dict for fast reference, deal with inconsistency between NodeWorldID and NodeJournalID
    for  journal_id, file_id, file_name in zip(node_file['JournalNodeID'], node_file['WorldFileID'], node_file['WorldFileName']):
        node_dict[journal_id] = (file_id,file_name)

    #result file, which follow the old ref node format
    base_ref_node_file = csv.writer(open('{}{}/reference/provenancefiltering/{}-provenancefiltering-base-ref-node.csv'.format(root_dir, dataset_name, dataset_index),
        'wb'), delimiter='|')
    base_ref_node_file.writerow(['ProvenanceProbeFileID','WorldFileID','WorldFileName','JournalNodeID'])

    donor_ref_node_file = csv.writer(open('{}{}/reference/provenancefiltering/{}-provenancefiltering-donor-ref-node.csv'.format(root_dir,dataset_name,dataset_index),
        'wb'), delimiter='|')
    donor_ref_node_file.writerow(['ProvenanceProbeFileID','WorldFileID','WorldFileName','JournalNodeID'])

    inter_ref_node_file = csv.writer(open('{}{}/reference/provenancefiltering/{}-provenancefiltering-inter-ref-node.csv'.format(root_dir,dataset_name,dataset_index),
        'wb'), delimiter='|')
    inter_ref_node_file.writerow(['ProvenanceProbeFileID','WorldFileID','WorldFileName','JournalNodeID'])

    final_ref_node_file = csv.writer(open('{}{}/reference/provenancefiltering/{}-provenancefiltering-final-ref-node.csv'.format(root_dir,dataset_name,dataset_index),
        'wb'), delimiter='|')
    final_ref_node_file.writerow(['ProvenanceProbeFileID','WorldFileID','WorldFileName','JournalNodeID'])

    for probe_id, journal_file_name in tqdm(zip(probe_index_file["ProvenanceProbeFileID"], probe_index_file['JournalFileName'])):
        try:
            with open('{}{}/{}'.format(root_dir, dataset_name, journal_file_name)) as json_file:    
                json_data = json.load(json_file)
        except:
            print('Error to read {}'.format('%s.json' % (journal_file_name)))
            continue
        
        node_data=json_data['nodes']
        
        base_index_list = []
        inter_index_list = []
        donor_index_list = []
        final_index_list = []

        for i in range(len(node_data)): 
            if node_data[i]['nodetype']=='base':
                base_index_list.append(i)
            if node_data[i]['nodetype']=='final':
                final_index_list.append(i)
            if node_data[i]['nodetype']=='donor':
                donor_index_list.append(i)
            if node_data[i]['nodetype']=='interim':
                inter_index_list.append(i)

        provenanceProbeFileID = probe_id

        for base_node_index in base_index_list:
            baseFileID = node_data[base_node_index]['file'][:-4]
            try:
                base_ref_node_file.writerow([provenanceProbeFileID,node_dict[baseFileID][0],node_dict[baseFileID][1],baseFileID])
            except:
                pass

        for final_node_index in final_index_list:
            finalFileID = node_data[final_node_index]['file'][:-4]
            try:
                final_ref_node_file.writerow([provenanceProbeFileID,node_dict[finalFileID][0],node_dict[finalFileID][1],finalFileID])
            except:
                pass

        for donor_node_index in donor_index_list:
            donorFileID = node_data[donor_node_index]['file'][:-4]
            try:
                donor_ref_node_file.writerow([provenanceProbeFileID,node_dict[donorFileID][0],node_dict[donorFileID][1],donorFileID])
            except:
                pass

        for inter_node_index in inter_index_list:
            interFileID = node_data[inter_node_index]['file'][:-4]
            try:
                inter_ref_node_file.writerow([provenanceProbeFileID,node_dict[interFileID][0],node_dict[interFileID][1],interFileID])
            except:
                pass
