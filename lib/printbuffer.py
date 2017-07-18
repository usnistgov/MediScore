"""
 *File: printbuffer.py
 *Date: 07/13/2017
 *Author: Daniel Zhou
 *Status: Complete

 *Description: this code contains a print buffer to be used to print output to console where
               multiprocessing is involved.


 *Disclaimer:
 This software was developed at the National Institute of Standards
 and Technology (NIST) by employees of the Federal Government in the
 course of their official duties. Pursuant to Title 17 Section 105
 of the United States Code, this software is not subject to copyright
 protection and is in the public domain. NIST assumes no responsibility
 whatsoever for use by other parties of its source code or open source
 server, and makes no guarantees, expressed or implied, about its quality,
 reliability, or any other characteristic."
"""

class printbuffer:
    """
    This class aggregates verbose printout for verbose atomic printout
    """
    def __init__(self,verbose):
        self.verbose = verbose
        self.s=[]

    def append(self,mystring):
        if self.verbose == 1:
            self.s.append(mystring)

    def atomprint(self,lock):
        if self.verbose == 1:
            self.s.append("================================================================================")
            with lock:
                print('\n'.join(self.s))

