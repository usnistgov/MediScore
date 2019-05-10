import os
import Render as p
from collections import OrderedDict

"""
Description: a function using the Render package to plot a default ROC curve.
"""
#for use with detection metrics plotter
class detPackage:
    def __init__(self,
                 tpr,
                 fpr,
                 fpr_stop,
                 ci_tpr,
                 auc,
                 nTarget,
                 nNonTarget):
        """
        This class is a wrapper for a collection of metrics for rendering the ROC curve
        of the detection metrics.
        """
        self.tpr = tpr
        self.fpr = fpr
        self.fpr_stop = fpr_stop
        self.ci_tpr = ci_tpr
        self.auc = auc
        self.t_num = nTarget
        self.nt_num = nNonTarget
        self.d = None #TODO: possibility of computing d in the future
    
def plotROC(mydets,plotname,plot_title,outdir):
    #initialize plot options for ROC
    plot_opts = OrderedDict([
            ('title', plot_title),
            ('subtitle', ''),
            ('plot_type', 'ROC'),
            ('title_fontsize', 13),  # 15
            ('subtitle_fontsize', 11),
            ('xticks_size', 'medium'),
            ('yticks_size', 'medium'),
            ('xlabel', "False Alarm Rate [%]"),
            ('xlabel_fontsize', 11),
            ('ylabel', "Miss Detection Rate [%]"),
            ('ylabel_fontsize', 11)])

    opts_list = [OrderedDict([('color', 'red'),
                              ('linestyle', 'solid'),
                              ('marker', '.'),
                              ('markersize', 6),
                              ('markerfacecolor', 'red'),
                              ('label',None),
                              ('antialiased', 'False')])]

    #compute AUC and EER with detection metrics and store in
    #add ci_tpr to rocvalues
#            rocvalues['ci_tpr'] = 0

    configRender = p.setRender([mydets],opts_list,plot_opts)
    myRender = p.Render(configRender)
    myroc = myRender.plot_curve()

    #save roc curve in the output. Automatically closes the plot.
    #To deal with RuntimeError involving plots
    count = 0
    while True:
        try:
            myroc.savefig(os.path.join(outdir,'%s.pdf' % plotname), bbox_inches='tight')
            break
        except RuntimeError:
            count += 1
            print("Attempt {} failed. Attempting to save figure again...".format(count))

    return myroc

