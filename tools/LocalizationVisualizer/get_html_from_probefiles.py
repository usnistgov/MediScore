import sys
import os
import pandas as pd
#from skimage.transform import resize
import cv2
import argparse
libdir=os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../lib')
sys.path.append(libdir)
import masks

tablestring="<h2>{}</h2><br/>\
{}<br/>\
<table border='1'>\
    <tbody>\
        <tr>\
            <td>\
            {}x{}\
            </td>\
            <td>\
            {}x{}\
            </td>\
        </tr>\
        <tr>\
            <td>\
            <img src='{}' alt='probe file' style='width:{}px;'>\
            </td>\
            <td>\
            <img src='{}' alt='mask file' style='width:{}px;'>\
            </td>\
        </tr>\
    </tbody>\
</table>"
#            <img src='{}' alt='mask file' style='width:{}px;'>

def gen_directory(task,output_root,probeids):
    outdir=os.path.join(output_root,'_'.join(probeids))
    os.system('mkdir -p {}'.format(outdir))
    if task == 'splice':
        os.system('mkdir -p {}/probe'.format(outdir))
        os.system('mkdir -p {}/donor'.format(outdir))
    return outdir

def link_to_directories(dfrow,origdir,outdir):
    task = dfrow['TaskID']
    if task == 'manipulation':
        probedir = dfrow['ProbeFileID']
    elif task == 'splice':
        probedir = "_".join([dfrow["ProbeFileID"],dfrow["DonorFileID"]])
    origdirfull = os.path.join(origdir,probedir)
    outdirfull = os.path.join(outdir,probedir)
    os.system('rm -f {}'.format(outdirfull))
    os.symlink(origdirfull,outdirfull)
    return dfrow

def scale_image(matrix,compress=100,inplace=False):
    if compress == 100:
        return matrix
    dims = matrix.shape
    new_dims = (dims[1]*compress/100,dims[0]*compress/100)
    new_matrix = cv2.resize(matrix,new_dims)

    if inplace:
        matrix = new_matrix
        return
    else:
        return new_matrix

def gen_overlay(probeimgname,maskimgname,outdir,compress=100,outfile="overlay.png",newcolor=[255,0,0]):
    maskImg = masks.refmask_color(maskimgname)
    probeImg = masks.mask(probeimgname)

    #change blacks to red if black is the only other color in the mask
    if set(maskImg.getUniqueValues()) <= {0,(1 << 24) - 1}:
        print("Reassigning color...")
        maskImg.matrix[maskImg.matrix[:,:,0] == 0] = newcolor[::-1]

    overlaydir = os.path.join(outdir,outfile)
    try:
        print("Beginning overlay of {} on {}...".format(maskimgname,probeimgname))
        overlayimg = maskImg.overlay(probeimgname)
        print("Writing overlay...")
        overlayimg = scale_image(overlayimg,compress)
        cv2.imwrite(overlaydir,overlayimg,[16,0])
    except:
        probefileext = probeimgname.split('.')[-1]
        print("Overlay of {} on {} not possible with OpenCV. Respective dimensions are: {} and {}. Defaulting to probefile.{}.".format(maskimgname,probeimgname,maskImg.get_dims(),probeImg.get_dims(),probefileext))
        os.system('rm -f {}'.format(overlaydir))
        os.symlink(os.path.join(outdir,'probefile.{}'.format(probefileext)),overlaydir)

    return overlaydir

def link_or_save(sourcename,destname,compress=100,overwrite=False):
    imgsfx = sourcename.split('.')[-1]
    if (not os.path.isfile(destname)) and ((compress==100) or (imgsfx=='jp2')):
        os.symlink(sourcename,destname)
    elif (not os.path.isfile(destname)) or (os.path.isfile(destname) and overwrite):
        sourceImg = masks.mask(sourcename,readopt=1)
        sourceImg.matrix = scale_image(sourceImg.matrix,compress)
        os.system('rm -f {}'.format(destname))
        cv2.imwrite(destname,sourceImg.matrix)

def gen_html_page(dfrow,refdir,outdir,compress=100,overwrite=False):
    task = dfrow['TaskID']
    #mkdir
    if task == 'manipulation':
#        modes = ['Probe']
        modes = ['Probe','ProbeBitPlane']
        probeids = [dfrow['ProbeFileID']]
        page_title = "ProbeFileID: {}".format(probeids[0])
    elif task == 'splice':
#        modes = ['Probe','Donor']
        modes = ['Probe','Donor','BinaryProbe']
        probeids = [dfrow['ProbeFileID'],dfrow['DonorFileID']]
        page_title = "ProbeFileID: {}, DonorFileID: {}".format(probeids[0],probeids[1])

    page_dir = gen_directory(task,outdir,probeids)
    page_name = '_'.join(probeids)

    #iterate over the mask files
    for m in modes:
        file_name_clause = "{}MaskFileName: {}".format(m,dfrow['%sMaskFileName' % m])

        page_out_dir = page_dir
        if m == 'Donor':
            h_ref_field = 'DonorHeight'
            w_ref_field = 'DonorWidth'
            probefile = 'DonorFileName'
            page_out_dir = os.path.join(page_dir,'donor')
        else:
            h_ref_field = 'ProbeHeight'
            w_ref_field = 'ProbeWidth'
            probefile = 'ProbeFileName'
            if task == 'splice':
                page_out_dir = os.path.join(page_dir,'probe')

        #create symbolic links
        probefilename = os.path.join(refdir,dfrow[probefile])
        probefileext = probefilename.split('.')[-1]
        probefilelink = os.path.join(page_out_dir,'probefile.{}'.format(probefileext))
        
        display_scale_factor = max(1,int(dfrow[w_ref_field]/640))
        probe_display_width = int(dfrow[w_ref_field]/display_scale_factor)
        try:
            link_or_save(probefilename,probefilelink,compress,overwrite)
        except:
            print("Image {} could not be compressed and saved in a smaller size. Attempting to link to original file.".format(probefilename))
            os.system("rm -f {}".format(probefilelink))
            link_or_save(probefilename,probefilelink,100,overwrite)

        maskname = os.path.join(refdir,dfrow['%sMaskFileName' % m])
        masksfx = maskname.split('.')[-1]
        masklink = os.path.join(page_out_dir,'{}mask.{}'.format(m.lower(),masksfx))
        mask_display_width = int(dfrow['%sMaskWidth' % m]/display_scale_factor)
        os.system("rm -f {}".format(masklink))
        link_or_save(maskname,masklink,compress,overwrite)

        if m in ['Probe','Donor']:
            gen_overlay(probefilelink,masklink,page_out_dir,compress)

        #make page content for probes and donors
        page_content = tablestring.format(page_title,file_name_clause,\
                                          dfrow[h_ref_field],dfrow[w_ref_field],\
                                          dfrow['%sMaskHeight' % m],dfrow['%sMaskWidth' % m],\
                                          os.path.basename(probefilelink),probe_display_width,\
                                          os.path.basename(masklink),mask_display_width)
        #write the html to page_out_dir
        myhtml = open(os.path.join(page_out_dir,'%s.html' % m.lower()),'w+')
        myhtml.write(page_content)
        myhtml.close()

    return dfrow

def paint_cols_red(df,cols):
    """
    Note: cols is a list of lists of column values that ought to be equal
    """
    #if any columns are equal, paint bold red
    prefix = '<p style="color:red"><b>'
    postfix = '</b></p>'

    eqlist = []
    for c in cols:
        df[c] = df[c].astype(str)
        eqlist = []
        for i,elt in enumerate(c[1:]):
            eqlist.append("({}=={})".format(c[i],c[i+1]))
#    eqlist = [ "({}=={})".format(c[i],c[i+1]) for i,elt in enumerate(c[1:]) for c in cols]
        neq_query = " ".join(["not","(" + " & ".join(eqlist) + ")"])
        neq_idx = df.query(neq_query).index
        df.loc[neq_idx,c] = prefix + df.loc[neq_idx,c] + postfix
    return df

def gen_home_page(df,outdir,reduce_redundancy=False):
    #gen home page pointing to other pages
    task = df['TaskID'].iloc[0]
    df_out = df.copy()
    pd.set_option('display.max_colwidth',-1)
    
    if task == 'manipulation':
#        modes = ['Probe']
        modes = ['Probe','ProbeBitPlane']
    elif task == 'splice':
#        modes = ['Probe','Donor']
        modes = ['Probe','Donor','BinaryProbe']

    #assign links to each page
    shrink_w=150
    for m in modes:
        file_field = '%sMaskFileName' % m
        if m == 'Donor':
            html_dirs = df_out['ProbeFileID'] + '_' + df_out['DonorFileID'] + '/donor'
        else:
            if task == 'splice':
                html_dirs = df_out['ProbeFileID'] + '_' + df_out['DonorFileID'] + '/probe'
            elif task == 'manipulation':
                html_dirs = df_out['ProbeFileID']
            
        #image links, with 150 px width to start off
        imgclause = '<img src="' + html_dirs + '/{}mask.png" width="{}" border="1" alt="{}img">'.format(m.lower(),shrink_w,m.lower())
#        df_out['%sMaskImage' % m] = '<a href="' + html_dirs + '/%s.html">' % m.lower() + imgclause + '</a>'
        if reduce_redundancy and (m in ['Probe','Donor']):
            id_field = "%sFileID" % m
            file_field = "%sFileName" % m
            mask_field = "%sMaskFileName" % m
            id_clause = "<b>" + id_field + "</b>: " +  df_out[id_field]
            file_clause = "<b>" + file_field + "</b>: " + df_out[file_field]
            mask_clause = "<b>" + mask_field + "</b>: " + df_out[mask_field]
            df_out[m] = id_clause + '<br/>' + file_clause + '<br/>' + mask_clause + '<br/>' +\
                        '<br/><a href="' + html_dirs + '/%s.html">' % m.lower() + '<img src="' + html_dirs + '/overlay.png" width="{}" border="1" alt="{}img"></a>'.format(shrink_w,m.lower())
#            df_out['%sFileName' % m] = df_out["%sFileName" % m] + '<br/><a href="' + html_dirs + '/%s.html">' % m.lower() + '<img src="' + html_dirs + '/overlay.png" width="{}" border="1" alt="probeimg"></a>'.format(shrink_w)
        else:
            if m in ['Probe','Donor']:
    #            df_out['%sImage' % m] = '<img src="' + html_dirs + '/overlay.png" width="{}" border="1" alt="probemask">'.format(shrink_w)
                df_out['%sFileName' % m] = df_out["%sFileName" % m] + '<br/><img src="' + html_dirs + '/overlay.png" width="{}" border="1" alt="{}mask">'.format(shrink_w,m.lower())
            if m == "ProbeBitPlane":
                df_out['%sMaskFileName' % m] = '<br/><a href="' + html_dirs + '/%s.html">' % m.lower() + df_out["%sMaskFileName" %m] + '</a>'
            else:
                df_out['%sMaskFileName' % m] = df_out["%sMaskFileName" % m] + '<br/><a href="' + html_dirs + '/%s.html">' % m.lower() + imgclause + '</a>'

    #reorder df_out columns
    firstcols = ["Dataset",'TaskID','JournalName']
    if task == 'manipulation':
        addcols = ['Probe','ProbeBitPlaneMaskFileName'] if reduce_redundancy else ['ProbeFileID','ProbeFileName','ProbeMaskFileName','ProbeBitPlaneMaskFileName']
#        addcols = ['ProbeFileName','ProbeImage','ProbeMaskFileName','ProbeMaskImage','ProbeBitPlaneMaskFileName']
        wcol_list = [['ProbeWidth','ProbeMaskWidth','ProbeBitPlaneMaskWidth']]
        hcol_list = [['ProbeHeight','ProbeMaskHeight','ProbeBitPlaneMaskHeight']]
        valid_cols = ['ProbeValid','Comments']
    elif task == 'splice':
        addcols = ['Probe','BinaryProbeMaskFileName','Donor'] if reduce_redundancy else ['ProbeFileID','ProbeFileName','ProbeMaskFileName','BinaryProbeMaskFileName','DonorFileID','DonorFileName','DonorMaskFileName']
#        addcols = ['ProbeFileName','ProbeImage','ProbeMaskFileName','ProbeMaskImage','BinaryProbeMaskFileName','BinaryProbeMaskImage','DonorFileID','DonorFileName','DonorImage','DonorMaskFileName','DonorMaskImage']
        wcol_list = [['ProbeWidth','ProbeMaskWidth','BinaryProbeMaskWidth'],['DonorWidth','DonorMaskWidth']]
        hcol_list = [['ProbeHeight','ProbeMaskHeight','BinaryProbeMaskHeight'],['DonorHeight','DonorMaskHeight']]
        valid_cols = ['ProbeValid','DonorValid','Comments']

    #set of mistmatched dimensions. Paint them all red.
    for wcol in wcol_list:
        paint_cols_red(df_out,wcol_list)
    for hcol in hcol_list:
        paint_cols_red(df_out,hcol_list)

    whcols = []
    for i,l in enumerate(wcol_list):
        whcols.extend(l)
        whcols.extend(hcol_list[i])

    if not set(valid_cols) < set(df_out.columns.values.tolist()):
        valid_cols = []

    col_order = firstcols + addcols + whcols + valid_cols
    df_out = df_out[col_order]
        
    home_page = open(os.path.join(outdir,'homepage.html'),'w+')
    home_page.write(df_out.to_html(escape=False,na_rep='').replace("text-align: right;","text-align: center;").encode('utf-8'))
    home_page.close()

    return df_out

def partition_df(df,N):
    if N == 0:
        return [df]
    nrow = df.shape[0]
    df_list = [df.iloc[k:k+N] for k in range(0,nrow,N)]
    return df_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate HTML pages for discrepant images.")
    parser.add_argument('-t','--task',type=str,default='manipulation',
        help="Two different types of tasks: [manipulation] and [splice]",metavar='character')
    parser.add_argument('--refDir',type=str,
        help='Dataset directory path: [e.g., ../../data/NC2016_Test]',metavar='valid directory')
    parser.add_argument('-rx','--refIndex',type=str,
        help="The ref merged with a chosen index, generated by get_dims_from_probes.py",metavar='valid csv file')
    parser.add_argument('-oR','--outRoot',type=str,
        help="Output root of the HTML pages",metavar='valid file path')
    parser.add_argument('-soR','--sourceOutRoot',type=str,default=None,
        help="Original output root of the HTML pages. Primarily important when generating subsets of the HTML pages. Default: argument passed to --outRoot",metavar='valid_file_path')
    parser.add_argument('-N',type=int,default=0,
        help="The number of probes per homepage partition. 0 will output all the rows to one homepage; numbers greater than 0 will be output to separate directories suffixed by '_k', where k is the number of the partition in order of ProbeFileID.",metavar="integer")
    parser.add_argument('-hp','--homepage_only',action='store_true',
        help="Generate the homepage only. Should be used only if the individual pages are all generated. If used with -N, will also generate symbolic links to the generated probes in the original directory.")
    parser.add_argument('--compress',type=int,default=100,
        help="Save images scaled to a percentile factor. For example, passing a factor of 20 will scale the image to 20% of its original dimensions. If used with -hp, will override -hp. Default: 100.",metavar="integer scale factor")
    parser.add_argument('-rr','--reduce_redundancy',action='store_true',
        help="Do not generate a probe mask image or donor mask image for the overall HTML, due to existence of overlaid images.")
    parser.add_argument('-ow','--overwrite',action='store_true',
        help="Overwrite existing images if they exist.")
    parser.add_argument('--filter',action='store_true',
        help="Filter out the rows for which no badness has been detected.")
    
    if len(sys.argv) < 2:
        parser.print_help()
        exit(0)
    args = parser.parse_args()
    if args.sourceOutRoot is None:
        args.sourceOutRoot = args.outRoot

    refdir = os.path.abspath(args.refDir)
    outdir = os.path.abspath(os.path.dirname(args.outRoot))
    sourceoutdir = os.path.abspath(os.path.dirname(args.sourceOutRoot))

    if args.task == 'manipulation':
        sortby=['JournalName','ProbeFileID']
    elif args.task == 'splice':
        sortby=['JournalName','ProbeFileID','DonorFileID']

    refIndex = pd.read_csv(args.refIndex,sep="|",header=0,na_filter=False)
    if args.filter:
        cols = refIndex.columns.values.tolist()
        if args.task == 'manipulation':
            validation_cols = ["ProbeValid"]
        elif args.task == 'splice':
            validation_cols = ["ProbeValid","DonorValid"]
        val_cols_all = validation_cols + ["Comments"]
        if set(val_cols_all) < set(cols):
            val_query = [ "(%s!='y')" % c for c in validation_cols ]
            val_query = " | ".join(val_query) + " | (Comments != '')"
            refIndex = refIndex.query(val_query)
    refIndex = refIndex.sort_values(sortby).reset_index(drop=True)

    dflist = partition_df(refIndex,args.N)
    is_multi_df = len(dflist) > 1

    for i,df in enumerate(dflist):
        output_dir='%s_%d' % (outdir,i) if is_multi_df else outdir
        df_out_name=os.path.join(output_dir,"homepage_%d.csv" % i) if is_multi_df else os.path.join(output_dir,"homepage.csv")
        os.system('mkdir -p {}'.format(output_dir))
        print("Proceeding to generate per-image HTML output for {}...".format(output_dir))
        if (not args.homepage_only) or (args.compress != 100):
            df.apply(gen_html_page,axis=1,refdir=refdir,outdir=output_dir,compress=args.compress,overwrite=args.overwrite)
        else:
            print("Per-image HTML output has been generated. No need to generate it further.")
            #generate symbolic links to directories
            if is_multi_df:
                df.apply(link_to_directories,axis=1,origdir=sourceoutdir,outdir=output_dir)
    
        print("HTML output generated. Proceeding to generate home page...")
        gen_home_page(df,output_dir,args.reduce_redundancy)
        df.to_csv(df_out_name,sep="|",index=False)
        print("Homepage generated.")
#        os.system('tar -chjf {} {}'.format("%s.tar.bz2" % output_dir,output_dir))

