import pandas as p
import numpy as np
import cv2
import sys
import os
import multiprocessing
from numpngw import write_apng
from string import Template
import numbers
lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../lib')
sys.path.append(lib_path)
from masks import refmask,refmask_color,mask
from constants import *
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except:
    print("Matplotlib failed to import. Threshold plots cannot be generated.")

def gen_img_pages(args):
    return html_generator.apply_gen_imgs(*args)

def is_number(n):
    if isinstance(n,numbers.Number):
        if not np.isnan(n):
            return True
    return False

debug_off=False

class html_generator():
    def __init__(self,
                 task,
                 perimage_df,
                 average_df,
                 journal_df,
                 refdir,
                 sysdir,
                 outroot,
                 query='',
                 overwrite=True,
                 usejpeg2000=False,
                 cache_dir=None):
        self.task = task
        self.perimage_df = perimage_df
        self.average_df = average_df
        self.journal_df = journal_df
        self.refdir = refdir
        self.sysdir = sysdir
        self.outroot = os.path.dirname(outroot)
        self.outpfx = os.path.basename(outroot)
        
        self.query = query
        self.mode = 0
        self.overwrite = overwrite
        self.usejpeg2000 = usejpeg2000
        self.cache_dir=cache_dir #NOTE: for now

    def gen_report(self,params):
        processors = params.processors
        self.kernel = params.kernel
        self.eks = params.eks
        self.dks = params.dks
        self.ntdks = params.ntdks
        self.nspx = params.nspx
        self.pppns = params.pppns

        manager = multiprocessing.Manager()
        self.msg_queue = manager.Queue()
        #dictionary of colors corresponding to confusion measures
        self.cols = {'tpcol':'green','fpcol':'red','tncol':'white','fncol':'blue','bnscol':'yellow','snscol':'pink','pnscol':'purple'}
        self.hexs = self.nums2hex(self.cols.values())

        if self.perimage_df.shape[0] > 0:
            if self.task == 'manipulation':
                self.mode = 0
                self.gen_perimage(processors)
            elif self.task == 'splice':
                for mode in [1,2]:
                    self.mode = mode
                    self.gen_perimage(processors)
        self.gen_top_page()

        default_name = 'html_mask_scores_perimage.csv'
        if self.overwrite:
            pi_name = '%s_mask_scores_perimage.csv' % self.outpfx
        else:
            pi_name = default_name
        self.perimage_df.to_csv(os.path.join(self.outroot,pi_name),sep="|",index=False)

        #output all messages
        while not self.msg_queue.empty():
            msg = self.msg_queue.get()
            print("*"*30)
            print(msg)
        return 0

    #define average report
    def gen_top_page(self):
        html_out = self.perimage_df.copy()
        #os.path.join doesn't seem to work with Pandas Series so just do a manual string addition
        if self.outroot[-1] == '/':
            self.outroot = self.outroot[:-1]

        pd.set_option('display.max_colwidth',-1)
        #set links around the system output data frame files for images that are not NaN
        
        if self.task == 'manipulation':
            score_pfx = ['']
        elif self.task == 'splice':
            score_pfx = ['Probe','Donor']

        for pfx in score_pfx:
            scored_name = '%sScored' % pfx
            link_idx = html_out[scored_name] == 'Y'
            if pfx == '':
                probe_pfx = 'Probe'
                html_dirs = html_out.ix[link_idx,'ProbeFileID']
            elif pfx == 'Probe':
                probe_pfx = 'Probe'
                html_dirs = html_out.ix[link_idx,'ProbeFileID'] + '_' + html_out.ix[link_idx,'DonorFileID'] + '/probe'
            elif pfx == 'Donor':
                probe_pfx = 'Donor'
                html_dirs = html_out.ix[link_idx,'ProbeFileID'] + '_' + html_out.ix[link_idx,'DonorFileID'] + '/donor'

            probe_file_field = '%sFileName' % probe_pfx
            html_out.loc[link_idx,probe_file_field] = '<a href="' + html_dirs + '/' \
                                                                  + html_out.ix[link_idx,probe_file_field].str.split('/').str.get(-1).str.split('.').str.get(0) + '.html">' \
                                                                  + html_out.ix[link_idx,probe_file_field].str.split('/').str.get(-1) + '</a>'

        #write to index.html
        fname = os.path.join(self.outroot,'index.html')
        myf = open(fname,'w')

        #add other metrics where relevant
        if self.average_df is not 0:
            if self.task == 'manipulation':
                met_pfx = ['']
            elif self.task == 'splice':
                met_pfx = ['p','d']

            #write title and then average_df
            for pfx in met_pfx:
                metriclist = {}
                for met in ['NMM','MCC','BWL1']:
                    metriclist['{}Optimum{}'.format(pfx,met)] = 3
                    if is_number(self.average_df['{}ActualThreshold'.format(pfx)]):
                        metriclist['{}Maximum{}'.format(pfx,met)] = 3
                        metriclist['{}Actual{}'.format(pfx,met)] = 3
                for met in ['GWL1','AUC','EER']:
                    metriclist[''.join([pfx,met])] = 3

            a_df_copy = self.average_df.copy().round(metriclist)

            myf.write('<h3>Average Scores</h3>\n')
            myf.write(a_df_copy.to_html().replace("text-align: right;","text-align: center;"))

        html_out = html_out.round(metriclist)

        #insert graphs here
        if self.task == 'manipulation':
            myf.write("<br/><table><tbody><tr><td><embed src=\"mask_average_roc.pdf\" alt=\"mask_average_roc\" width=\"540\" height=\"540\" type='application/pdf'></td><td><embed src=\"pixel_average_roc.pdf\" alt=\"pixel_average_roc\" width=\"540\" height=\"540\"></td></tr><tr><th>Mask Average ROC</th><th>Pixel Average ROC</th></tr></tbody></table><br/>\n")
        elif self.task == 'splice':
            myf.write("<br/><table><tbody><tr><td><embed src=\"mask_average_roc_probe.pdf\" alt=\"mask_average_roc_probe\" width=\"540\" height=\"540\" type='application/pdf'></td><td><embed src=\"pixel_average_roc_probe.pdf\" alt=\"pixel_average_roc_probe\" width=\"540\" height=\"540\" type='application/pdf'></td></tr><tr><th>Probe Average ROC</th><th>Probe Pixel Average ROC</th></tr><tr><td><embed src=\"mask_average_roc_donor.pdf\" alt=\"mask_average_roc_donor\" width=\"540\" height=\"540\" type='application/pdf'</td><td><embed src=\"pixel_average_roc_donor.pdf\" alt=\"pixel_average_roc_donor\" width=\"540\" height=\"540\" type='application/pdf'></td></tr><tr><th>Donor Average ROC</th><th>Donor Pixel Average ROC</th></tr></tbody></table><br/>\n")

        #write the query if manipulated
        if self.query != '':
            myf.write("\nFiltered by query: {}\n".format(self.query))

        myf.write('<h3>Per Scored Trial Scores</h3>\n')
        myf.write(html_out.to_html(escape=False,na_rep='').replace("text-align: right;","text-align: center;").encode('utf-8'))
        myf.write('\n')

        myf.close()

    def gen_perimage(self,processors):
        maxprocs = max(multiprocessing.cpu_count() - 2,1)
        #if more, print warning message and use max processors
        nrow = self.perimage_df.shape[0]
        if (processors > nrow) and (nrow > 0):
            print("Warning: too many processors for rows in the data. Defaulting to rows in data ({}).".format(nrow))
            processors = nrow
        if processors > maxprocs:
            print("Warning: the machine does not have that many processors available. Defaulting to max ({}).".format(maxprocs))
            processors = maxprocs

        if processors == 1:
            #case for one processor for efficient debugging and to eliminate overhead when running
            self.perimage_df = self.perimage_df.apply(self.gen_one_img_page,axis=1,reduce=False)
        else:
            #split self.perimage_df into array of dataframes based on number of processors (and rows in the file)
            chunksize = nrow//processors
            perimage_dfS = [[self,self.perimage_df[i:(i+chunksize)]] for i in range(0,nrow,chunksize)]

            p = multiprocessing.Pool(processes=processors)
            perimage_dfS = p.map(gen_img_pages,perimage_dfS)
            p.close()
            p.join()

            #re-merge in the order found and return
            self.perimage_df = pd.concat(perimage_dfS)

        if isinstance(self.perimage_df,pd.Series):
            self.perimage_df = self.perimage_df.to_frame().transpose()

    def apply_gen_imgs(self,df):
        return df.apply(self.gen_one_img_page,axis=1,reduce=False)

    def gen_one_img_page(self,pi_row):
        #self.manipReport(task,subOutRoot,manipFileID,manipFileName,baseFileName,rImg,sImg,rbin_name,sbinmaskname,smask_threshold,thresMets,bns,sns,pns,mets,mymeas,colMaskName,aggImgName,myprintbuffer)
        if self.mode == 0:
            scored_col = 'Scored'
        elif self.mode == 1:
            scored_col = 'ProbeScored'
        elif self.mode == 2:
            scored_col = 'DonorScored'
        #don't generate page for an image that wasn't scored
        if pi_row[scored_col] == 'N':
            return pi_row

        probe_ID_field = 'ProbeFileID'
        probe_file_field = 'ProbeFileName'
        probe_mask_file_field = 'ProbeMaskFileName'
        if self.usejpeg2000:
            probe_mask_file_field = 'ProbeBitPlaneMaskFileName'
        output_probe_mask_file_field = 'OutputProbeMaskFileName'
        mymode = 'Probe'

        printbuffer = []
        if self.mode == 0:
            output_dir = os.path.join(self.outroot,pi_row['ProbeFileID'])
        elif self.mode > 0:
            output_dir = os.path.join(self.outroot,'_'.join([pi_row['ProbeFileID'],pi_row['DonorFileID']]))
            if self.mode == 1:
                probe_mask_file_field = 'BinaryProbeMaskFileName'
                output_dir = os.path.join(output_dir,'probe')
            elif self.mode == 2:
                output_dir = os.path.join(output_dir,'donor')
                probe_ID_field = 'DonorFileID'
                probe_file_field = 'DonorFileName'
                probe_mask_file_field = 'DonorMaskFileName'
                output_probe_mask_file_field = 'OutputDonorMaskFileName'
                mymode = 'Donor'

        probe_file_id = pi_row[probe_ID_field]
        probe_file_name = pi_row[probe_file_field]
        probe_mask_file_name = pi_row[probe_mask_file_field]
        refpath = os.path.join(self.refdir,probe_mask_file_name)
        if probe_mask_file_name in [np.nan,'',None]:
            if self.usejpeg2000 and (self.mode == 0):
                refpath = os.path.join(self.sysdir,'whitemask_ref.jp2')
            else:
                refpath = os.path.join(self.sysdir,'whitemask_ref.png')

        output_probe_mask_file_name = pi_row[output_probe_mask_file_field]
        if output_probe_mask_file_name in [np.nan,'',None]:
            output_probe_mask_file_name = 'whitemask.png'

        #read in the weight images
        bns_name = os.path.join(output_dir,'_'.join([probe_file_id,'bns.png']))
        sns_name = os.path.join(output_dir,'_'.join([probe_file_id,'sns.png']))
        pns_name = os.path.join(output_dir,'_'.join([probe_file_id,'pns.png']))

        rImg = 0
        bwts,swts = 0,0
        if not (os.path.isfile(bns_name) and os.path.isfile(sns_name)):
            jData = self.journal_df.query("ProbeFileID=='{}'".format(probe_file_id))
            if self.usejpeg2000 and self.mode == 0:
                rImg = refmask(refpath,jData=jData,mode=self.mode)
            else:
                rImg = refmask_color(refpath,jData=jData,mode=self.mode)
            printbuffer.append("Generating weights image...")
            mywts,bwts,swts = rImg.aggregateNoScore(self.eks,self.dks,self.ntdks,self.kernel)

        #read in erode kern size, dilate kern size, and all that no-score jazz. Generate the files where present
        if bwts is 0:
            try:
                _,bwts=cv2.threshold(cv2.imread(bns_name,cv2.IMREAD_GRAYSCALE),0,1,cv2.THRESH_BINARY)
            except:
                None

        if swts is 0:
            try:
                _,swts=cv2.threshold(cv2.imread(sns_name,cv2.IMREAD_GRAYSCALE),0,1,cv2.THRESH_BINARY)
            except:
                None

        sImg = 0
        pwts = 0
        if not os.path.isfile(pns_name):
            sImg = mask(os.path.join(output_dir,output_probe_mask_file_name))
            no_score_pixel = self.nspx
            if self.pppns:
                try:
                    no_score_pixel = pi_row['{}OptOutPixelValue'.format(mymode)]
                except:
                    pass
            pwts = sImg.pixelNoScore(no_score_pixel)

        if pwts is 0:
            try:
                _,pwts=cv2.threshold(cv2.imread(pns_name,cv2.IMREAD_GRAYSCALE),0,1,cv2.THRESH_BINARY)
            except:
                None

        sys_base = os.path.basename(output_probe_mask_file_name)[:-4]

        color_weight_name = '-'.join([sys_base,'weights.png'])
        color_weight_path = os.path.join(output_dir,color_weight_name)
        
        mywts = cv2.bitwise_and(cv2.bitwise_and(bwts,swts),pwts)
        
        dims = bwts.shape
        colwts = 255*np.ones((dims[0],dims[1],3),dtype=np.uint8)
        colwts[bwts==0] = colordict['yellow']
        colwts[swts==0] = colordict['pink']
        colwts[pwts==0] = colordict['purple']

        printbuffer.append("Saving weights image...")
        cv2.imwrite(color_weight_path,colwts)

        display_width = min(dims[1],640) #limit on width of images for readability of the report

        #generate HTML files
        printbuffer.append("Reading HTML template...")
        html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../html_template.txt")
        with open(html_path,'r') as f:
            htmlstr = Template(f.read())

        jtable = ''
        if self.mode == 0:
            jtable = self.gen_jtable(probe_file_id,printbuffer)

        #generate the HTML for the metrics table
        met_table = self.gen_metrics_table(pi_row)

        thres_pfx = ''
        if self.mode == 1:
            thres_pfx = 'p'
        elif self.mode == 2:
            thres_pfx = 'd'
        optT = pi_row['%sOptimumThreshold' % thres_pfx]
        met_table_prefix=''
        if is_number(optT):
            optT = int(optT)
            met_table_prefix = 'Optimum Threshold: {}<br>'.format(optT)
            if is_number(pi_row['%sActualThreshold' % thres_pfx]):
                met_table_prefix = "<ul><li><b>Optimum Threshold</b>: {}</li>\
                                    <li><b>Maximum Threshold</b>: {}</li>\
                                    <li><b>Actual Threshold</b>: {}</li></ul><br>".format(optT,pi_row['%sMaximumThreshold' % thres_pfx],pi_row['%sActualThreshold' % thres_pfx])
        met_table = ''.join([met_table_prefix,met_table])

        printbuffer.append("Computing pixel count...")
        totalpx = mywts.sum()
        allpx = dims[0]*dims[1]
        _,invwts = cv2.threshold(mywts,0,1,cv2.THRESH_BINARY_INV)
        totalns = invwts.sum()
        totals = {}
        totals['BNS'] = pi_row['%sPixelBNS' % thres_pfx]
        totals['SNS'] = pi_row['%sPixelSNS' % thres_pfx]
        totals['PNS'] = pi_row['%sPixelPNS' % thres_pfx]
        totals['TNS'] = totalns
        
        percstrings = {}
        for m in ['TP','TN','FP','FN','BNS','SNS','PNS','TNS']:
            percstrings[m] = "np.nan"
        
        #set up dictionary for and generate confusion measures
        conf_measures = {}
        for m in ['TP','TN','FP','FN']:
            m_name = '{}OptimumPixel{}'.format(thres_pfx,m)
            max_m_name = '{}MaximumPixel{}'.format(thres_pfx,m)
            am_name = '{}ActualPixel{}'.format(thres_pfx,m)
            if totalpx > 0:
                percstrings[m] = '{0:.3f}'.format(float(pi_row[m_name])/totalpx)
            conf_measures[m_name] = int(pi_row[m_name])
            if is_number(pi_row[am_name]):
                conf_measures[am_name] = int(self.average_df[am_name])
                conf_measures[max_m_name] = int(self.average_df[max_m_name])
        conf_measures['TotalPixels'] = totalpx
        
        conf_table = self.gen_confusion_table(conf_measures)
        #recolor the conf_table measures
        conf_table = (conf_table.replace("TP: green",'<font style="color:#{}">TP: green</font>'.format(self.hexs[self.cols['tpcol']]))
                                .replace("FP: red",'<font style="color:#{}">FP: red</font>'.format(self.hexs[self.cols['fpcol']]))
                                .replace("TN: white",'<font style="color:#{}">TN: white</font>'.format(self.hexs[self.cols['tncol']]))
                                .replace("FN: blue",'<font style="color:#{}">FN: blue</font>'.format(self.hexs[self.cols['fncol']]))
                                )

        if allpx > 0:
            for m in ['BNS','SNS','PNS','TNS']:
                percstrings[m] = '{0:.3f}'.format(float(totals[m])/allpx)

        thres_string = ''
        plt_width = 540 #NOTE: custom value for plot sizes

        #read in thresMets.
        thres_string = self.gen_thres_plot(pi_row,output_dir,plt_width,printbuffer)

        #build symbolic links for existing files
        probe_path = os.path.join(self.refdir,probe_file_name)
        m_base = os.path.basename(probe_path)
        r_base = os.path.basename(refpath)
        s_base = os.path.basename(output_probe_mask_file_name)

        basehtml = ''
        b_base = ''
        mpfx = mymode.lower()
        if self.mode < 2:
            b_path = os.path.join(self.refdir,pi_row['BaseFileName'])
            b_base = os.path.basename(b_path)
            b_base_new = 'baseFile%s' % b_base[-4:]
            b_path_new = os.path.join(output_dir,b_base_new)
            try:
                os.remove(b_path_new)
            except OSError:
                None
            printbuffer.append("Creating link for base image %s" % b_base)
            os.symlink(os.path.abspath(b_path),b_path_new)
            basehtml="<img src={} alt='base image' style='width:{}px;'>".format(b_base_new,plt_width)

        r_path_new = os.path.join(output_dir,'refMask.png')
        s_path_new = os.path.join(output_dir,'sysMask.png')
        m_path_new = os.path.join(output_dir,''.join([mpfx,'File',m_base[-4:]]))

        for mypath in [r_path_new,s_path_new,m_path_new]:
            try:
                os.remove(mypath)
            except OSError:
                None
        
        #if color, create a symbolic link. Otherwise, create and save the refMask.png
        printbuffer.append("Creating link for reference mask %s..." % refpath)
        if not self.usejpeg2000:
            os.symlink(os.path.abspath(refpath),r_path_new)
        elif self.mode == 0:
            #cache dir. Has much more impact in the report generating stage.
            journalkeys = self.gen_journal_keys()
            jdata = self.journal_df.query("ProbeFileID=='{}' and Color!=''".format(probe_file_id))[journalkeys]
            if self.cache_dir:
                cache_acolor_path = os.path.join(self.cache_dir,'%s_acolor.png' % probe_file_id)
                acolor_in_cache = os.path.isfile(cache_acolor_path)
                if acolor_in_cache:
                    os.symlink(os.path.abspath(cache_acolor_path),r_path_new)
                else:
                    if not rImg is 0:
                        rImg = refmask(refpath,jData=jdata,mode=0)
                    ref_mask = rImg.getAnimatedMask()
                    write_apng(r_path_new,ref_mask,delay=600,use_palette=False)
            else:
                rImg = refmask(refpath,jData=jdata,mode=0)
                ref_mask = rImg.getAnimatedMask()
                write_apng(r_path_new,ref_mask,delay=600,use_palette=False)
            if self.cache_dir:
                cache_acolor_path = os.path.join(self.cache_dir,'%s_acolor.png' % probe_file_id)
                if not acolor_in_cache:
                    write_apng(r_path_new,ref_mask,delay=600,use_palette=False)

        printbuffer.append("Creating link for manipulated image {}".format(refpath))
        os.symlink(os.path.abspath(probe_path),m_path_new)
        printbuffer.append("Creating link for system output mask {}".format(output_probe_mask_file_name))
        os.symlink(os.path.abspath(os.path.join(self.sysdir,output_probe_mask_file_name)),s_path_new)

        #generate aggregate image
        colMaskName,aggImgName = self.aggregate_color_masks(probe_file_id,
                                                            probe_file_name,
                                                            refpath,
                                                            os.path.join(self.sysdir,output_probe_mask_file_name),
                                                            output_dir,
                                                            bwts,
                                                            swts,
                                                            pwts)
        pi_row['%sColMaskFileName' % thres_pfx] = colMaskName
        pi_row['%sAggMaskFileName' % thres_pfx] = aggImgName

        syspfx = ''
        if is_number(self.average_df['%sActualThreshold' % thres_pfx].iloc[0]):
            syspfx = 'Actual '

        rbin_name = os.path.join(output_dir,'-'.join([refpath.split('/')[-1][:-4],'bin.png']))
        sbin_name = os.path.join(output_dir,'{}-bin.png'.format(os.path.basename(output_probe_mask_file_name)[:-4]))
        sys_threshold = pi_row['%sOptimumThreshold' % thres_pfx]
        if is_number(pi_row['%sActualThreshold' % thres_pfx]):
            sbin_name = os.path.join(output_dir,'{}-actual_bin.png'.format(os.path.basename(output_probe_mask_file_name)[:-4]))
            sys_threshold = pi_row['%sActualThreshold' % thres_pfx]

        printbuffer.append("Writing HTML...")
        htmlstr = htmlstr.substitute({'probeName': pi_row[probe_file_field],
                                      'probeFname': "".join([mpfx,'File',m_base[-4:]]),#mBase,
                                      'width': plt_width,
                                      'baseName': b_base,
                                      'basehtml': basehtml,#mBase,
                                      'aggMask' : os.path.basename(aggImgName),
                                      'refMask' : 'refMask.png',#rBase,
                                      'sysMask' : 'sysMask.png',#sBase,
                                      'binRefMask' : os.path.basename(rbin_name),
                                      'binSysMask' : os.path.basename(sbin_name),
                                      'systh' : sys_threshold,
                                      'noScoreZone' : os.path.basename(color_weight_name),
                                      'colorMask' : os.path.basename(colMaskName),
                                      'met_table' : met_table,
#  				      'nmm' : round(metrics['NMM'],3),
#  				      'mcc' : round(metrics['MCC'],3),
#  				      'bwL1' : round(metrics['BWL1'],3),
#  				      'gwL1' : round(metrics['GWL1'],3),
                                      'syspfx' : syspfx,
                                      'totalPixels' : totalpx,
                                      'conftable':conf_table,
                                      'bns' : int(totals['BNS']),
                                      'sns' : int(totals['SNS']),
                                      'pns' : int(totals['PNS']),
                                      'tns' : int(totals['TNS']),
                                      'bnscol':self.cols['bnscol'],
                                      'snscol':self.cols['snscol'],
                                      'pnscol':self.cols['pnscol'],
                                      'bnshex':self.hexs[self.cols['bnscol']],
                                      'snshex':self.hexs[self.cols['snscol']],
                                      'pnshex':self.hexs[self.cols['pnscol']],
                                      'percbns':percstrings['BNS'],
                                      'percsns':percstrings['SNS'],
                                      'percpns':percstrings['PNS'],
                                      'perctns':percstrings['TNS'],
                                      'jtable':jtable,
                                      'th_table':thres_string,
                                      'roc_curve':'<embed src=\"roc.pdf\" alt=\"roc curve\" width=\"{}\" height=\"{}\" type=\'application/pdf\'>'.format(plt_width,plt_width)}) #add journal operations and set bg color to the html

        #print htmlstr
        fprefix=os.path.basename(probe_file_name)
        fprefix=fprefix.split('.')[0]
        fname=os.path.join(output_dir,'.'.join([fprefix,'html']))
        myhtml=open(fname,'w')
        myhtml.write(htmlstr)
        printbuffer.append("HTML page written.")
        myhtml.close()
        return pi_row
    
    def aggregate_color_masks(self,probe_file_id,probe_file_name,ref_mask_name,sys_mask_name,output_dir,bns,sns,pns):
        #can simply get the eroded image from this image
        rmat = cv2.imread(os.path.join(output_dir,'{}-bin.png'.format(os.path.basename(ref_mask_name)[:-4])),1)
        rmat_sum = rmat[:,:,0] + rmat[:,:,1] + rmat[:,:,2] #only possible colors are white, yellow, pink, purple, and black. A basic sum yields no ambiguity.
        eData = np.uint8(rmat_sum)

        smat = cv2.imread(os.path.join(output_dir,'{}-bin.png'.format(os.path.basename(sys_mask_name)[:-4])),0)
        _,b_sImg = cv2.threshold(smat,0,1,cv2.THRESH_BINARY_INV)
        _,b_eImg = cv2.threshold(eData,0,2,cv2.THRESH_BINARY_INV)
        _,b_bnsImg = cv2.threshold(bns,0,4,cv2.THRESH_BINARY_INV)
        _,b_snsImg = cv2.threshold(sns,0,8,cv2.THRESH_BINARY_INV)
        _,b_pnsImg = cv2.threshold(pns,0,16,cv2.THRESH_BINARY_INV)
        mImg = b_sImg + b_eImg + b_bnsImg + b_snsImg + b_pnsImg

        #overlays
        mydims = rmat.shape
        mycolor = 255*np.ones(mydims,dtype=np.uint8)

        #get colors through colordict
        mycolor[mImg==1] = colordict['red'] #only system (FP)
        mycolor[mImg==2] = colordict['blue'] #only erode image (FN) (the part that is scored)
        mycolor[mImg==3] = colordict['green'] #system and erode image coincide (TP)
        mycolor[(mImg>=4) & (mImg <=7)] = colordict['yellow'] #boundary no-score zone
        mycolor[(mImg>=8) & (mImg <=15)] = colordict['pink'] #selection no-score zone
        mycolor[mImg>=16] = colordict['purple'] #system opt out

        #return path to mask
        color_agg_mask_name = sys_mask_name.split('/')[-1]
        color_agg_mask_base = color_agg_mask_name.split('.')[0]
        final_mask_name = "%s_colored.jpg" % color_agg_mask_base
        color_agg_path=os.path.abspath(os.path.join(output_dir,final_mask_name))
        #write the aggregate mask to file
        cv2.imwrite(color_agg_path,mycolor)

        #also aggregate over the grayscale probe_file_name for direct comparison
        #save as animated png if not using color.
        probeImg = mask(os.path.join(self.refdir,probe_file_name))
        mData = probeImg.matrix
        m3chan = np.stack((mData,mData,mData),axis=2)
        myagg = np.copy(m3chan)

        if self.usejpeg2000 and (self.mode == 0):
            ref_mask = refmask(ref_mask_name,jData=self.journal_df.query("ProbeFileID=='{}'".format(probe_file_id)))
        else:
            ref_mask = refmask_color(ref_mask_name,jData=self.journal_df.query("ProbeFileID=='{}'".format(probe_file_id)),mode=self.mode)
        ref_mask.binarize(254)
        refbw = ref_mask.bwmat

        alpha=0.7
        #for modified images, weighted sum the colored mask with the grayscale
        #NOTE: try/catch the overlay error. Print out the shapes of all involved items.
        try:
            refmat = ref_mask.matrix
            composite_path = ''
            if not self.usejpeg2000:
                modified = cv2.addWeighted(refmat,alpha,m3chan,1-alpha,0)
                myagg[refbw==0] = modified[refbw==0]
                composite_mask_name = "_".join([color_agg_mask_base,"composite.jpg"])
                composite_path = os.path.join(output_dir,composite_mask_name)
                cv2.imwrite(composite_path,myagg)
            elif self.mode == 0:
                if self.cache_dir:
                    cache_acolor_path = os.path.join(os.path.join(self.cache_dir,'%s_acolorpart.npy' % probe_file_id))
                    if os.path.isfile(cache_acolor_path):
                        #get this animated mask saved in and read from the cache
                        aseq = np.load(cache_acolor_path)
                    else:
                        aseq = ref_mask.getAnimatedMask('partial')
                else:
                    aseq = ref_mask.getAnimatedMask('partial')
                seq = []
                for frame in aseq:
                    #join the frame with the grayscale manipulated image
                    refmat = frame
                    modified = cv2.addWeighted(frame,alpha,m3chan,1-alpha,0)
                    layermask = (frame[:,:,0] != 255) | (frame[:,:,1] != 255) | (frame[:,:,2] != 255)
                    aggfr = np.copy(m3chan)
                    #overlay colors with particular manipulated regions
                    aggfr[layermask] = modified[layermask]
                    seq.append(aggfr)
                composite_mask_name = "_".join([color_agg_mask_base,"composite.png"])
                composite_path = os.path.abspath(os.path.join(output_dir,composite_mask_name))
                write_apng(composite_path,seq,delay=600,use_palette=False)
                if self.cache_dir:
                    cache_acolor_path = os.path.join(os.path.join(self.cache_dir,'%s_acolorpart.npy' % probe_file_id))
                    if not os.path.isfile(cache_acolor_path):
                        np.save(cache_acolor_path,aseq)
        except:
            exc_type,exc_obj,exc_tb = sys.exc_info()
            print("Exception {} encountered at line {} during mask overlay. Reference mask shape: {}. Probe image shape: {}".format(exc_type,exc_tb.tb_lineno,refmat.shape,m3chan.shape))
#            return {'mask':path,'agg':''}
            return color_agg_path,''

        return color_agg_path,composite_path

    def gen_journal_keys(self):
        evalcol='Evaluated'
        journal_headers = self.journal_df.columns.values.tolist()
        journalkeys = ['Operation','Purpose','Color',evalcol]
        if self.usejpeg2000:
            journalkeys.insert(0,'BitPlane')
        if 'Sequence' in journal_headers:
            journalkeys.insert(0,'Sequence')
        return journalkeys

    def gen_jtable(self,probe_file_id,printbuffer):
        printbuffer.append("Composing journal table...")

        journalkeys = self.gen_journal_keys()
        jdata = self.journal_df.query("ProbeFileID=='{}' and Color!=''".format(probe_file_id))[journalkeys]
        if 'Sequence' in self.journal_df.columns.values.tolist():
            jdata = jdata.sort_values("Sequence",ascending=False)

        jdata_color_list = list(jdata['Color'])
        jdata_col_arrays = [c.split(' ')[::-1] for c in jdata_color_list] #reverse order for compatability with how it is stored in a matrix
        jdata_col_arrays = [[int(x) for x in c] for c in jdata_col_arrays]
        
        jdata_hex = pd.Series([self.num2hex(c) for c in jdata_col_arrays],index=jdata.index) #match the indices
        jdata['Color'] = 'td bgcolor="#' + jdata_hex + '"btd'
        jtable = jdata.to_html(index=False)
        jtable = jtable.replace('<td>td','<td').replace('btd','>') #substitute the characters in
        return jtable

    def gen_thres_plot(self,pi_row,output_dir,plt_width,printbuffer):
        if self.mode == 0:
            thres_pfx = ''
        elif self.mode == 1:
            thres_pfx = 'p'
        elif self.mode == 2:
            thres_pfx = 'd'

        thres_mets = pd.read_csv(os.path.join(output_dir,'thresMets.csv'),sep="|",header=0)
        if thres_mets.dropna().shape[0] > 1:
            printbuffer.append("Generating MCC per threshold graph...")
            #plot MCC
            try:
                plotname = 'thresMets.png'
                plt.plot(thres_mets['Threshold'],thres_mets['MCC'],'bo',thres_mets['Threshold'],thres_mets['MCC'],'k')
                #plot cyan point for supremum, purple for maximum and red point for actual if sbin >= 0, and legend with two or three as appropriate
                optT = pi_row['%sOptimumThreshold' % thres_pfx]
                optMCC = pi_row['%sOptimumMCC' % thres_pfx]
                optpt, = plt.plot([optT],[optMCC],'co',markersize=12)
                handles = [optpt]
                labels = ['Optimum MCC']
                actT = pi_row['%sActualThreshold' % thres_pfx]
                if is_number(actT):
                    maxT = pi_row['%sMaximumThreshold' % thres_pfx]
                    maxMCC = pi_row['%sMaximumMCC' % thres_pfx]
                    maxpt, = plt.plot([maxT,maxMCC],'mo',markersize=8)
                    handles.append(maxpt)
                    labels.append('Maximum MCC')
                
                    actMCC = pi_row['%sActualMCC' % thres_pfx]
                    actpt, = plt.plot([actT],[actMCC],'ro',markersize=8)
                    handles.append(actpt)
                    labels.append('Actual MCC')
                #move legend elsewhere if it obstructs points. (e.g., OptimumThreshold > 200, for estimate)
                legend_loc='upper right'
                if optT > 200:
                    legend_loc='upper left'
                plt.legend(handles,labels,loc=legend_loc,borderaxespad=0,prop={'size':8},shadow=True,fontsize='small',numpoints=1)
                plt.suptitle('MCC per Threshold',fontsize=14)
                plt.xlabel("Binarization threshold value")
                plt.ylabel("Matthews Correlation Coefficient (MCC)")
                thres_string = os.path.join(output_dir,plotname)
                plt.savefig(thres_string,bbox_inches='tight') #save the graph
                plt.close()
                thres_string = "<img src=\"{}\" alt=\"thresholds graph\" style=\"width:{}px;\">".format(plotname,plt_width)
            except:
                e = sys.exc_info()[0]
                print("Warning: The plotter encountered error: {}. Defaulting to table display for the HTML report.".format(e))
                thres_mets = thres_mets.round({'NMM':3,
                                               'MCC':3,
                                               'BWL1':3,
                                               'GWL1':3,
                                               'AUC':3,
                                               'EER':3})
                thres_string = '<h4>Measures for Each Threshold</h4><br/>' + thres_mets.to_html(index=False).replace("text-align: right;","text-align: center;")
        else:
            thres_string = 'Threshold graph not applicable<br>'

        return thres_string

    def nums2hex(self,colors):
        """
        * Description: this function outputs the hexadecimal strings for a dictionary of colors for the HTML report
        * Inputs:
        *     colors: list of strings corresponding to the colors in self.colordict
        * Outputs:
        *     hexcolors: dictionary of hexadecimal color codes
        """
        hexcolors = {}
        for c in colors:
            mybgr = colordict[c]
            hexcolors[c] = self.num2hex(mybgr)
        return hexcolors

    def num2hex(self,color):
        """
        * Description: this function converts one BGR color to a hex string at a time.
                       Colors are reversed to conform with OpenCV's handling of colors.
        * Inputs:
        *     color: a list of three integers from 0 to 255 denoting a color code
        * Outputs:
        *     hexcolor: the hexadecimal color code corresponding to that color
        """

        myb = hex(color[0])[2:]
        myg = hex(color[1])[2:]
        myr = hex(color[2])[2:]
        if len(myb)==1:
            myb = '0%s' % myb 
        if len(myg)==1:
            myg = '0%s' % myg 
        if len(myr)==1:
            myr = '0%s' % myr
        hexcolor = (''.join([myr,myg,myb])).upper()
        return hexcolor

    def gen_metrics_table(self,metrics,mets_for_some=['NMM','MCC','BWL1'],mets_for_all=['GWL1','AUC','EER'],
                          rename_dict={'NMM':'Nimble Mask Metric (NMM)',
                                       'MCC':'Matthews Correlation Coefficient (MCC)',
                                       'BWL1':'Binary Weighted L1 Loss (BWL1)',
                                       'GWL1':'Grayscale Weighted L1 Loss (GWL1)',
                                       'AUC':'Area Under (ROC) Curve (AUC)',
                                       'EER':'Equal Error Rate'}):
        """
        *Description: this function generates the HTML string for the table of metrics and is not meant
                      to be used otherwise

        * Inputs:
        *    metrics: dictionary or series of the metrics to be scored
        *    mets_for_some: list of metrics to evaluated that differ for different thresholds
        *    mets_for_all: list of metrics that do not differ for different thresholds
        *    rename_dict: dictionary of new names for metrics. Keys must be the union of mets_for_some
                          and mets_for_all

        * Output
        *    tablestring: the html string for the generated table
        """
        met_table = pd.DataFrame(index=mets_for_some,columns=['Optimum'])
        #probe/donor
        actual_name = 'ActualThreshold'
        met_pfx = ''
        if self.mode == 1:
            met_pfx = 'p'
            actual_name = 'p%s' % actual_name
        elif self.mode == 2:
            met_pfx = 'd'
            actual_name = 'd%s' % actual_name
        actual_thres = is_number(self.average_df[actual_name].iloc[0])
        
        for met in mets_for_some:
            metstr = "nan"
            #round if numeric
            optmetname = '{}Optimum{}'.format(met_pfx,met)
            if (not isinstance(metrics[optmetname],str)) and not np.isnan(metrics[optmetname]):
                metstr = "{0:.3f}".format(metrics[optmetname])

            met_table.at[met,'Optimum'] = metstr
            if actual_thres:
                for met_type in ['Maximum','Actual']:
                    ametstr = "nan"
                    amet_name = ''.join([met_pfx,met_type,met])
                    if not isinstance(metrics[amet_name],str):
                        ametstr = "{0:.3f}".format(metrics[amet_name])
                    met_table.at[met,met_type] = ametstr

        #rename indices
        rename_keys = rename_dict.keys()
        sub_rename_dict = {m:rename_dict[m] for m in mets_for_some if m in rename_keys}
        met_table.rename(index=sub_rename_dict,inplace=True)

        tablestring = met_table.to_html(index=True).replace("text-align: right;","text-align: center;")
        otherrows = ''
        colspan = 1
        #make column span the row
        if actual_thres:
            colspan = 3
        for met in mets_for_all:
            #add into each row
            metname = ''.join([met_pfx,met])
            if not isinstance(metrics[metname],str) and not np.isnan(metrics[metname]):
                metstr = "{0:.3f}".format(metrics[metname])
            otherrows = '\n'.join([otherrows,"<tr><th>{}</th><td colspan={}>{}</td></tr>".format(rename_dict[met],colspan,metstr)])
        #tack onto end of met_table.rename
        tablestring = (tablestring.replace("</tbody>","".join([otherrows,"</tbody>"]))
                                  .replace("<th></th>","<th>Localization Metrics</th>")
                                  .replace("<th>","<th align='left'>")
                                  .replace("<tr>","<tr align='right'>"))
        
        
        return tablestring

    def gen_confusion_table(self,conf_metrics,mets_for_some=['TP','FP','TN','FN'],mets_for_all=[],
                          rename_dict={'TP':'True Postives (TP: green)',
                                       'FP':'False Postives (FP: red)',
                                       'TN':'True Negatives (TN: white)',
                                       'FN':'False Negatives (FN: blue)'}):
        """
        *Description: this function generates the HTML string for the table of confusion measures (TP, TN, FP, and FN)
                      and is not meant to be used otherwise
        """
        #probe/donor
        actual_name = 'ActualThreshold'
        met_pfx = ''
        if self.mode == 1:
            met_pfx = 'p'
            actual_name = 'p%s' % actual_name
        elif self.mode == 2:
            met_pfx = 'd'
            actual_name = 'd%s' % actual_name

        actual_thres = is_number(self.average_df[actual_name].iloc[0])
        
        met_table = pd.DataFrame(index=mets_for_some,columns=['OptimumPixelCount','OptimumProportion'])
        totalpx = conf_metrics['TotalPixels']
        for met in mets_for_some:
            metstr = "nan"
            #generate Pixel count and Proportion separation
            optcol = '{}OptimumPixel{}'.format(met_pfx,met)
            met_table.at[met,'OptimumPixelCount'] = conf_metrics[optcol]

            #round if numeric
            if totalpx > 0:
                metstr = "{0:.3f}".format(float(conf_metrics[optcol])/totalpx)
            met_table.at[met,'OptimumProportion'] = metstr

            #do the same for Actual and Maximum metrics
            if actual_thres:
                for met_type in ['Maximum','Actual']:
                    ametstr = "nan"
                    optcol = '{}{}Pixel{}'.format(met_pfx,met_type,met)
                    met_table.at[met,'%sPixelCount' % met_type] = conf_metrics[optcol]
    
                    if totalpx > 0:
                        ametstr = "{0:.3f}".format(float(conf_metrics[optcol])/totalpx)
                    met_table.at[met,'%sProportion'] = ametstr

        #rename indices
        rename_keys = rename_dict.keys()
        sub_rename_dict = {m:rename_dict[m] for m in mets_for_some if m in rename_keys}
        met_table.rename(index=sub_rename_dict,inplace=True)
        cols = ['OptimumPixelCount','OptimumProportion']
        met_table.OptimumPixelCount = met_table.OptimumPixelCount.astype(int)
        if actual_thres:
            for met_type in ['Maximum','Actual']:
                cols.extend(['%sPixelCount' % met_type,'%sProportion' % met_type])
                met_table.at['%sPixelCount' % met_type] = met_table['%sPixelCount'].astype(int)
        met_table = met_table[cols]

        tablestring = met_table.to_html(index=True).replace("text-align: right;","text-align: center;")
        otherrows = ''
        for met in mets_for_all:
            #add into each row
            metname = "{}{}".format(met_pfx,met)
            if not isinstance(conf_metrics[metname],str):
                metstr = "{0:.3f}".format(metrics[metname])
            otherrows = '\n'.join([otherrows,"<tr><th>{}</th><td>{}</td></tr>".format(rename_dict[met],metstr)])
        #tack onto end of met_table.rename
        tablestring = (tablestring.replace("</tbody>","".join([otherrows,"</tbody>"]))
                                  .replace("<th></th>","<th>Confuson Measures</th>")
                                  .replace("<th>","<th align='left'>")
                                  .replace("<tr>","<tr align='right'>")
                                  .replace('<table border="1" class="dataframe">','<table border="1" class="dataframe" bgcolor="#C8C8C8">'))

        return tablestring

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate a graphical report for the localization masks and their scores.')
    parser.add_argument('-t','--task',type=str,default='manipulation',
        help='Two different types of tasks: [manipulation] and [splice]',metavar='character')
    parser.add_argument('-pi','--perimage',type=str,
        help="The file of per-image localization scores generated by the localization scorer.",metavar='valid file path')
    parser.add_argument('-avg','--average',type=str,
        help="The file of average localization scores generated by the localization scorer.",metavar='valid file path')
    parser.add_argument('-j','--journal',type=str,
        help="The file of journal manipulations evaluated, generated by the localization scorer.",metavar='valid file path')
    parser.add_argument('--refDir',type=str,
        help="Dataset directory path: [e.g., ../../data/test_suite/maskscorertests]",metavar='valid directory string')
    parser.add_argument('--sysDir',type=str,
        help="System output directory path: [e.g., ../../data/NC2016_Test].",metavar='valid directory string')
    parser.add_argument('-oR','--outRoot',type=str,
        help="The directory where the localization scores are located, plus file output prefix. This is also where the report pages will be produced.",metavar="valid directory string")
    parser.add_argument('-qm','--queryManipulation',default='',
        help="The query passed to -qm in localization scoring.",metavar="query string, according to pandas.query syntax")
    parser.add_argument('--overwrite',action='store_true',
        help="Update the original perimage file with the names of the color mask and aggregate image illustrating the scoring run for that image")
    parser.add_argument('--jpeg2000',action='store_true',
        help="Whether or not the JPEG2000 was used in the localization scoring run.")

    parser.add_argument('-k','--kernel',type=str,default='box',
        help="Convolution kernel type for erosion and dilation. Choose from [box],[disc],[diamond],[gaussian], or [line]. The default is 'box'.",metavar='character')
    parser.add_argument('--eks',type=int,default=15,
        help="Erosion kernel size number must be odd, [default=15]",metavar='integer')
    parser.add_argument('--dks',type=int,default=11,
        help="Dilation kernel size number must be odd, [default=11]",metavar='integer')
    parser.add_argument('--ntdks',type=int,default=15,
        help="Non-target dilation kernel for distraction no-score regions. Size number must be odd, [default=15]",metavar='integer')
    parser.add_argument('--nspx',type=int,default=-1,
        help="Set a pixel value for all system output masks to serve as a no-score region [0,255]. -1 indicates that no particular pixel value will be chosen to be the no-score zone. [default=-1]",metavar='integer')
    parser.add_argument('-pppns','--perProbePixelNoScore',action='store_true',
        help="Use the pixel values in the ProbeOptOutPixelValue column (DonorOptOutPixelValue as well for the splice task) of the system output to designate no-score zones. This value will override the value set for the global no-score pixel.")

    parser.add_argument('-p','--processors',type=int,default=1,
        help="The number of processors to use in the computation. Choosing too many processors will cause the program to forcibly default to a smaller number. [default=1].",metavar='integer')
    args = parser.parse_args()
    
    perimage_df = pd.read_csv(args.perimage,sep="|",header=0,na_filter=False)
    average_df = pd.read_csv(args.average,sep="|",header=0,na_filter=False)
    journal_df = pd.read_csv(args.journal,sep="|",header=0,na_filter=False)

    html_gen = html_generator(task=args.task,
                              perimage_df = perimage_df,
                              average_df = average_df,
                              journal_df = journal_df,
                              refdir = args.refDir,
                              sysdir = args.sysDir,
                              outroot = args.outRoot,
                              query = args.queryManipulation,
                              overwrite = args.overwrite,
                              usejpeg2000 = args.jpeg2000,
                              cache_dir=None
    )

    class params:
        def __init__(self,**kwds):
            self.__dict__.update(kwds)

    loc_params = params(kernel = args.kernel,
                        eks = args.eks,
                        dks = args.dks,
                        ntdks = args.ntdks,
                        nspx = args.nspx,
                        pppns = args.perProbePixelNoScore,
                        processors = args.processors)
    html_gen.gen_report(loc_params)
