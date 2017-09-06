"""
* File: validatorUnitTest.py
* Date: 11/18/2016
* Translation by: Daniel Zhou
* Original Author: Yooyoung Lee
* Status: Complete 

* Description: This object tests the functions of the unit test. 

* Requirements: This code requires the following packages:
    - cv2
    - numpy
    - pandas
    - unittest
  
  The rest should be available on your system.

* Inputs
    * -x, inIndex: index file name
    * -s, inSys: system output file name
    * -vt, valType: validator type: SSD or DSD
    * -v, verbose: Control printed output. -v 0 suppresss all printed output except argument parsng errors, -v 1 to print all output.

* Outputs
    * 0 if the files are validly formatted, 1 if the files are not.

* Disclaimer: 
This software was developed at the National Institute of Standards 
and Technology (NIST) by employees of the Federal Government in the
course of their official duties. Pursuant to Title 17 Section 105 
of the United States Code, this software is not subject to copyright 
protection and is in the public domain. NIST assumes no responsibility 
whatsoever for use by other parties of its source code or open source 
server, and makes no guarantees, expressed or implied, about its quality, 
reliability, or any other characteristic."
"""

import sys
import os
import cv2
import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod
import unittest as ut
import contextlib
from validator import SSD_Validator,DSD_Validator

identify=False
NCID='NC2016'
neglectMask=False
procs = 10

identify_string = ''
if identify:
    identify_string = '-id '
nm_string = ''
if neglectMask:
    nm_string = '-nm '


##want to take in a command and return the printed output to a string
#def print_capture(command):
#    #change stdout to another file
#    sys.stdout = open("validator.log","w")
#    val=exec(command)
#    mystr = 0
#    with open("validator.log","r") as myfile:
#        mystr = myfile.readlines()
#    mystr = "\n".join(mystr)    
#
#    #change back to stdout
#    sys.stdout = sys.__stdout__
#    return val,mystr

def msgcapture(fname):
    with open(fname,"r") as myfile:
        mystr = myfile.readlines()
        mystr = "\n".join(mystr)
        return mystr
    

class TestValidator(ut.TestCase):

    def testSSDName(self):
        import StringIO
        #This code taken from the following site: http://stackoverflow.com/questions/14197009/how-can-i-redirect-print-output-of-a-function-in-python
        @contextlib.contextmanager
        def stdout_redirect(where):
            sys.stdout = where
            try:
                yield where
            finally:
                sys.stdout = sys.__stdout__
        validatorRoot = '../../data/test_suite/validatorTests/'
        global verbose
        verbose = 1

        print("BASIC FUNCTIONALITY validation of SSDValidator beginning...")
#        myval = SSD_Validator(validatorRoot + 'foo_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_1/foo_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv')
#        self.assertEqual(myval.fullCheck(True,identify,NCID,neglectMask),0)
        myval = os.system("python2 validator.py -nc --ncid {} -vt SSD -s {} -x {} -p {} {}{}> vmb.log".format(NCID,validatorRoot + 'foo_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_1/foo_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv',procs,identify_string,nm_string))//256 
        self.assertEqual(myval,0)
        print("BASIC FUNCTIONALITY validated.")
        os.system('rm vmb.log')
        
        print("\nBeginning experiment ID naming error validations. Expect ERROR printouts for the next couple of cases. This is normal here.")
        print("CASE 0: Validating behavior when files don't exist.")
        
#        myval = SSD_Validator(validatorRoot + 'emptydir_NC2016_UnitTest_Splice_ImgOnly_p-baseline_1/foo__NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-manipulation-index0.csv')
        myval = os.system("python2 validator.py -nc --ncid {} -vt SSD -s {} -x {} -p {} {}{}> vm0.log".format(NCID,validatorRoot + 'emptydir_NC2016_UnitTest_Splice_ImgOnly_p-baseline_1/foo__NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-manipulation-index0.csv',procs,identify_string,nm_string))//256 
        self.assertEqual(myval,1)
        errstr = msgcapture("vm0.log")
        
#        with stdout_redirect(StringIO.StringIO()) as errmsg:
#            result=myval.fullCheck(True,identify,NCID,neglectMask) 
#        errmsg.seek(0)
#        errstr,val = print_capture('myval.fullCheck(True,identify,NCID,neglectMask)')
#        self.assertEqual(val,1)
#        errstr = errmsg.read() #NOTE: len(errmsg.read())==0, but when you set it equal, you get the entire string. What gives?
        self.assertTrue("ERROR: I can't find your system output" in errstr)
#        errmsg.close()
        
        print("CASE 0 validated.")
        os.system('rm vm0.log')
        
        print("\nCASE 1: Validating behavior when detecting consecutive underscores ('_') in name...")
#        myval = SSD_Validator(validatorRoot + 'foo__NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_1/foo__NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv')
        myval = os.system("python2 validator.py -nc --ncid {} -vt SSD -s {} -x {} -p {} {}{}> vm1.log".format(NCID,validatorRoot + 'foo__NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_1/foo__NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv',procs,identify_string,nm_string))//256 
        errstr = msgcapture("vm1.log")
        
#        with stdout_redirect(StringIO.StringIO()) as errmsg:
#            result=myval.fullCheck(True,identify,NCID,neglectMask)
#        errmsg.seek(0)
#        errstr = errmsg.read()

#        errstr,val = print_capture('myval.fullCheck(True,identify,NCID,neglectMask)')
        self.assertEqual(myval,1)
        self.assertTrue("ERROR: What kind of task is" in errstr)
        print("CASE 1 validated.")
        os.system('rm vm1.log')
        
        print("\nCASE 2: Validating behavior when detecting excessive underscores elsewhere...")
#        myval = SSD_Validator(validatorRoot + 'fo_o_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_1/fo_o_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv')
        myval = os.system("python2 validator.py -nc --ncid {} -vt SSD -s {} -x {} -p {} {}{}> vm2.log".format(NCID,validatorRoot + 'fo_o_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_1/fo_o_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv',procs,identify_string,nm_string))//256 
        errstr = msgcapture("vm2.log")
        
#        with stdout_redirect(StringIO.StringIO()) as errmsg:
#            result=myval.fullCheck(True,identify,NCID,neglectMask)
#        errmsg.seek(0)
#        errstr,val = print_capture('myval.fullCheck(True,identify,NCID,neglectMask)')
        self.assertEqual(myval,1)
#        errstr = errmsg.read()
        self.assertTrue("ERROR: What kind of task is" in errstr)
        print("CASE 2 validated.")
        os.system('rm vm2.log')
        
#        print("\nCASE 3: Validating behavior when detecting '+' in file name and an unrecognized task...")
#        myval = SSD_Validator(validatorRoot + 'foo+_NC2016_UnitTest_Manip_ImgOnly_p-baseline_1/foo+_NC2016_UnitTest_Manip_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv')
#        
#        with stdout_redirect(StringIO.StringIO()) as errmsg:
#            result=myval.fullCheck(True,identify,NCID,neglectMask)
#        errmsg.seek(0)
#        self.assertEqual(result,1)
#        errstr = errmsg.read()
#        self.assertTrue("ERROR: The team name must not include characters" in errstr)
#        self.assertTrue("ERROR: What kind of task is" in errstr)
#        print("CASE 3 validated.")
 
    def testSSDContent(self):
        import StringIO
        @contextlib.contextmanager
        def stdout_redirect(where):
            sys.stdout = where
            try:
                yield where
            finally:
                sys.stdout = sys.__stdout__
        validatorRoot = '../../data/test_suite/validatorTests/'
        global verbose
        verbose = None
        print("Validating syntactic content of system output.\nCASE 4: Validating behavior for incorrect headers and different number of rows than in index file...")
        print("CASE 4a: Validating behavior for incorrect headers")
#        myval = SSD_Validator(validatorRoot + 'foo_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_2/foo_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_2.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv') 
        myval = os.system("python2 validator.py -nc --ncid {} -vt SSD -s {} -x {} -p {} {}{}> vm4a.log".format(NCID,validatorRoot + 'foo_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_2/foo_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_2.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv',procs,identify_string,nm_string))//256 
        errstr = msgcapture("vm4a.log")
#        with stdout_redirect(StringIO.StringIO()) as errmsg:
#            result=myval.fullCheck(True,identify,NCID,neglectMask)
#        errmsg.seek(0)
#        errstr,val = print_capture('myval.fullCheck(True,identify,NCID,neglectMask)')
#        errstr = errmsg.read()
        self.assertEqual(myval,1)
        self.assertTrue("ERROR: The required column" in errstr)
        os.system('rm vm4a.log')

        print("CASE 4b: Validating behavior for duplicate rows and different number of rows than in index file...")
#        myval = SSD_Validator(validatorRoot + 'foob_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_2/foob_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_2.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv')
        myval = os.system("python2 validator.py -nc --ncid {} -vt SSD -s {} -x {} -p {} {}{}> vm4b.log".format(NCID,validatorRoot + 'foob_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_2/foob_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_2.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv',procs,identify_string,nm_string))//256 
        errstr = msgcapture("vm4b.log")
#        with stdout_redirect(StringIO.StringIO()) as errmsg:
#            result=myval.fullCheck(True,identify,NCID,neglectMask)
#        errmsg.seek(0)
#        errstr,val = print_capture('myval.fullCheck(True,identify,NCID,neglectMask)')
#        errstr = errmsg.read()
        self.assertEqual(myval,1)
        self.assertTrue("ERROR: Your system output contains duplicate rows" in errstr)
        self.assertTrue("ERROR: The number of rows in your system output (6) does not match the number of rows in the index file (5)." in errstr)
        os.system('rm vm4b.log')
        print("CASE 4 validated.")
        
        print("\nCASE 5: Validating behavior when mask is not a png...")
#        myval = SSD_Validator(validatorRoot + 'bar_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_1/bar_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv')
        myval = os.system("python2 validator.py -nc --ncid {} -vt SSD -s {} -x {} -p {} {}{}> vm5.log".format(NCID,validatorRoot + 'bar_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_1/bar_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv',procs,identify_string,nm_string))//256 
        errstr = msgcapture("vm5.log")
        
#        with stdout_redirect(StringIO.StringIO()) as errmsg:
#            result=myval.fullCheck(True,identify,NCID,neglectMask)
#        errmsg.seek(0)
#        errstr,val = print_capture('myval.fullCheck(True,identify,NCID,neglectMask)')
#        errstr = errmsg.read()
        self.assertEqual(myval,1)
        self.assertTrue("is not a png. Make it into a png!" in errstr)
        print("CASE 5 validated.")
        os.system('rm vm5.log')
        
        print("\nCASE 6: Validating behavior when mask is not single channel and when mask does not have the same dimensions.")
#        myval = SSD_Validator(validatorRoot + 'baz_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_1/baz_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv')
        myval = os.system("python2 validator.py -nc --ncid {} -vt SSD -s {} -x {} -p {} {}{}> vm6.log".format(NCID,validatorRoot + 'baz_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_1/baz_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv',procs,identify_string,nm_string))//256 
        errstr = msgcapture("vm6.log")
        
#        with stdout_redirect(StringIO.StringIO()) as errmsg:
#            result=myval.fullCheck(True,identify,NCID,neglectMask)
#        errmsg.seek(0)
#        errstr,val = print_capture('myval.fullCheck(True,identify,NCID,neglectMask)')
#        errstr = errmsg.read()
        self.assertEqual(myval,1)
        self.assertEqual(errstr.count("Dimensions"),2)
        self.assertTrue("ERROR: The mask image's length and width do not seem to be the same as the base image's." in errstr)
        self.assertTrue("is not single-channel." in errstr)
        print("CASE 6 validated.")
        os.system('rm vm6.log')
        
        print("\nCASE 7: Validating behavior when system output column number is less than 3.") 
#        myval = SSD_Validator(validatorRoot + 'foo_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_3/foo_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_3.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv')
        myval = os.system("python2 validator.py -nc --ncid {} -vt SSD -s {} -x {} -p {} {}{}> vm7.log".format(NCID,validatorRoot + 'foo_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_3/foo_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_3.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv',procs,identify_string,nm_string))//256 
        errstr = msgcapture("vm7.log")
        
#        with stdout_redirect(StringIO.StringIO()) as errmsg:
#            result=myval.fullCheck(True,identify,NCID,neglectMask)
#        errmsg.seek(0)
#        errstr,val = print_capture('myval.fullCheck(True,identify,NCID,neglectMask)')
#        errstr = errmsg.read()
        self.assertEqual(myval,1)
        self.assertTrue("ERROR: The number of columns of the system output file must be at least" in errstr)
        print("CASE 7 validated.")
        os.system('rm vm7.log')
        
        print("\nCASE 8: Validating behavior when mask file is not present.") 
#        myval = SSD_Validator(validatorRoot + 'foo_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_4/foo_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_4.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv')
        myval = os.system("python2 validator.py -nc --ncid {} -vt SSD -s {} -x {} -p {} {}{}> vm8.log".format(NCID,validatorRoot + 'foo_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_4/foo_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_4.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv',procs,identify_string,nm_string))//256 
        errstr = msgcapture("vm8.log")
        
#        with stdout_redirect(StringIO.StringIO()) as errmsg:
#            result=myval.fullCheck(True,identify,NCID,neglectMask)
#        errmsg.seek(0)
#        errstr,val = print_capture('myval.fullCheck(True,identify,NCID,neglectMask)')
#        errstr = errmsg.read()
        self.assertEqual(myval,1)
        self.assertTrue("does not exist! Did you name it wrong?" in errstr)
        
        print("CASE 8 validated.")
        os.system('rm vm8.log')
        
        print("\nALL SSD VALIDATION TESTS SUCCESSFULLY PASSED.")
                
    def testDSDName(self):
        import StringIO
        @contextlib.contextmanager
        def stdout_redirect(where):
            sys.stdout = where
            try:
                yield where
            finally:
                sys.stdout = sys.__stdout__
        validatorRoot = '../../data/test_suite/validatorTests/'
        global verbose
        verbose = None
        
        print("BASIC FUNCTIONALITY validation of DSDValidator.py beginning...")
        myval = os.system("python2 validator.py -nc --ncid {} -vt DSD -s {} -x {} -p {} {}{}> vsb.log".format(NCID,validatorRoot + 'lorem_NC2016_UnitTest_Splice_ImgOnly_p-baseline_1/lorem_NC2016_UnitTest_Splice_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv',procs,identify_string,nm_string))//256 
#        myval = DSD_Validator(validatorRoot + 'lorem_NC2016_UnitTest_Splice_ImgOnly_p-baseline_1/lorem_NC2016_UnitTest_Splice_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv')
#        self.assertEqual(myval.fullCheck(True,identify,NCID,neglectMask,validatorRoot + 'NC2016_Test0516_dfz/reference/splice/NC2016-splice-ref.csv'),0)
        self.assertEqual(myval,0)
        print("BASIC FUNCTIONALITY validated.")
        os.system('rm vsb.log')
        
        errmsg = ""
        #Same checks as Validate SSD, but applied to different files
        print("\nBeginning experiment ID naming error validations. Expect ERROR printouts for the next couple of cases. This is normal here.")
        print("\nCASE 0: Validating behavior when files don't exist.") 
#        myval = DSD_Validator(validatorRoot + 'emptydir_NC2016_UnitTest_Splice_ImgOnly_p-baseline_1/emptydir_NC2016_UnitTest_Splice_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-splice-index0.csv')
        myval = os.system("python2 validator.py -nc --ncid {} -vt DSD -s {} -x {} -p {} {}{}> vs0.log".format(NCID,validatorRoot + 'emptydir_NC2016_UnitTest_Splice_ImgOnly_p-baseline_1/emptydir_NC2016_UnitTest_Splice_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-splice-index0.csv',procs,identify_string,nm_string))//256 
        errstr = msgcapture("vs0.log")
#        with stdout_redirect(StringIO.StringIO()) as errmsg:
#            result=myval.fullCheck(True,identify,NCID,neglectMask)
#        errmsg.seek(0)
#        errstr = errmsg.read()
#        errstr,val = print_capture('myval.fullCheck(True,identify,NCID,neglectMask)')
        self.assertEqual(myval,1)
        self.assertTrue("ERROR: I can't find your system output" in errstr)
        self.assertTrue("ERROR: I can't find your index file" in errstr)
        print("CASE 0 validated.")
        os.system('rm vs0.log')
        
        print("\nCASE 1: Validating behavior when detecting consecutive underscores ('_') in name...")
#        myval = DSD_Validator(validatorRoot + 'lorem__NC2016_UnitTest_Spl_ImgOnly_p-baseline_1/lorem__NC2016_UnitTest_Spl_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv')
        myval = os.system("python2 validator.py -nc --ncid {} -vt DSD -s {} -x {} -p {} {}{}> vs1.log".format(NCID,validatorRoot + 'lorem__NC2016_UnitTest_Spl_ImgOnly_p-baseline_1/lorem__NC2016_UnitTest_Spl_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv',procs,identify_string,nm_string))//256 
        errstr = msgcapture("vs1.log")
#        with stdout_redirect(StringIO.StringIO()) as errmsg:
#            result=myval.fullCheck(True,identify,NCID,neglectMask)
#        errmsg.seek(0)
#        errstr = errmsg.read()
#        errstr,val = print_capture('myval.fullCheck(True,identify,NCID,neglectMask)')
        self.assertEqual(myval,1)
        self.assertTrue("ERROR: What kind of task is" in errstr)
        print("CASE 1 validated.")
        os.system('rm vs1.log')
        
        print("\nCASE 2: Validating behavior when detecting excessive underscores elsewhere...")
#        myval = DSD_Validator(validatorRoot + 'lor_em_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_1/lor_em_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv')
        myval = os.system("python2 validator.py -nc --ncid {} -vt DSD -s {} -x {} -p {} {}{}> vs2.log".format(NCID,validatorRoot + 'lor_em_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_1/lor_em_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv',procs,identify_string,nm_string))//256 
        errstr = msgcapture("vs2.log")
#        with stdout_redirect(StringIO.StringIO()) as errmsg:
#            result=myval.fullCheck(True,identify,NCID,neglectMask)
#        errmsg.seek(0)
#        errstr = errmsg.read()
#        errstr,val = print_capture('myval.fullCheck(True,identify,NCID,neglectMask)')
        self.assertEqual(myval,1)
        self.assertTrue("ERROR: What kind of task is" in errstr)
        print("CASE 2 validated.")
        os.system('rm vs2.log')
        
#        print("\nCASE 3: Validating behavior when detecting '+' in file name and an unrecogized task...\n")
#        myval = DSD_Validator(validatorRoot + 'lorem+_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_1/lorem+_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv')
#        with stdout_redirect(StringIO.StringIO()) as errmsg:
#            result=myval.fullCheck(True,identify,NCID,neglectMask)
#        errmsg.seek(0)
#        self.assertEqual(result,1)
#        errstr = errmsg.read()
#        self.assertTrue("ERROR: The team name must not include characters" in errstr)
#        self.assertTrue("ERROR: What kind of task is" in errstr)
#        print("CASE 3 validated.")
     
    def testDSDContent(self):
        import StringIO
        @contextlib.contextmanager
        def stdout_redirect(where):
            sys.stdout = where
            try:
                yield where
            finally:
                sys.stdout = sys.__stdout__
        validatorRoot = '../../data/test_suite/validatorTests/'
        global verbose
        verbose = None
        print("Validating syntactic content of system output.\nCASE 4: Validating behavior for incorrect headers, duplicate rows, and different number of rows than in index file...")
        print("CASE 4a: Validating behavior for incorrect headers, duplicate rows, and different number of rows than in index file...")
 #       self.assertEqual(myval,0)
#        myval = DSD_Validator(validatorRoot + 'lorem_NC2016_UnitTest_Splice_ImgOnly_p-baseline_2/lorem_NC2016_UnitTest_Splice_ImgOnly_p-baseline_2.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv')
        myval = os.system("python2 validator.py -nc --ncid {} -vt DSD -s {} -x {} -p {} {}{}> vs4a.log".format(NCID,validatorRoot + 'lorem_NC2016_UnitTest_Splice_ImgOnly_p-baseline_2/lorem_NC2016_UnitTest_Splice_ImgOnly_p-baseline_2.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv',procs,identify_string,nm_string))//256 
        errstr = msgcapture("vs4a.log")
#        with stdout_redirect(StringIO.StringIO()) as errmsg:
#            result=myval.fullCheck(True,identify,NCID,neglectMask)
#        errmsg.seek(0)
#        errstr = errmsg.read()
#        errstr,val = print_capture('myval.fullCheck(True,identify,NCID,neglectMask)')
        self.assertEqual(myval,1)
        self.assertTrue("ERROR: The required column" in errstr)
        os.system('rm vs4a.log')

        print("CASE 4b: Validating behavior for duplicate rows and different number of rows than in index file...")
#        myval = DSD_Validator(validatorRoot + 'loremb_NC2016_UnitTest_Splice_ImgOnly_p-baseline_2/loremb_NC2016_UnitTest_Splice_ImgOnly_p-baseline_2.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv')
        myval = os.system("python2 validator.py -nc --ncid {} -vt DSD -s {} -x {} -p {} {}{}> vs4b.log".format(NCID,validatorRoot + 'loremb_NC2016_UnitTest_Splice_ImgOnly_p-baseline_2/loremb_NC2016_UnitTest_Splice_ImgOnly_p-baseline_2.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv',procs,identify_string,nm_string))//256 
        errstr = msgcapture("vs4b.log")
#        with stdout_redirect(StringIO.StringIO()) as errmsg:
#            result=myval.fullCheck(True,identify,NCID,neglectMask)
#        errmsg.seek(0)
#        errstr = errmsg.read()
#        errstr,val = print_capture('myval.fullCheck(True,identify,NCID,neglectMask)')
        self.assertEqual(myval,1)
#        self.assertTrue("ERROR: Row" in errstr) #TODO: temporary measure until we get duplicates back
        self.assertTrue("ERROR: The number of rows in your system output (6) does not match the number of rows in the index file (5)." in errstr)
        os.system('rm vs4b.log')
        print("CASE 4 validated.")
        
#        print("\nCase 5: Validating behavior when the number of columns in the system output is less than 6.")
#        myval = DSD_Validator(validatorRoot + 'lorem_NC2016_UnitTest_Splice_ImgOnly_p-baseline_4/lorem_NC2016_UnitTest_Splice_ImgOnly_p-baseline_4.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv')
#        with stdout_redirect(StringIO.StringIO()) as errmsg:
#            result=myval.fullCheck(True,identify,NCID,neglectMask)
#        errmsg.seek(0)
#        self.assertEqual(result,1)
#        errstr = errmsg.read()
#        self.assertTrue("ERROR: The number of columns of the system output file must be at least" in errstr)
#        print("CASE 5 validated.")
        
        print("\nCASE 6: Validating behavior for mask semantic deviations. NC2016-1893.jpg and NC2016_6847-mask.jpg are (marked as) jpg's. NC2016_1993-mask.png is not single-channel. NC2016_4281-mask.png doesn't have the same dimensions...")
#        myval = DSD_Validator(validatorRoot + 'ipsum_NC2016_UnitTest_Splice_ImgOnly_p-baseline_1/ipsum_NC2016_UnitTest_Splice_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv')
        myval = os.system("python2 validator.py -nc --ncid {} -vt DSD -s {} -x {} -p {} {}{}> vs6.log".format(NCID,validatorRoot + 'ipsum_NC2016_UnitTest_Splice_ImgOnly_p-baseline_1/ipsum_NC2016_UnitTest_Splice_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv',procs,identify_string,nm_string))//256 
        errstr = msgcapture("vs6.log")
#        with stdout_redirect(StringIO.StringIO()) as errmsg:
#            result=myval.fullCheck(True,identify,NCID,neglectMask)
#        errmsg.seek(0)
#        errstr = errmsg.read()
#        errstr,val = print_capture('myval.fullCheck(True,identify,NCID,neglectMask)')
        self.assertEqual(myval,1)
        self.assertTrue("is not a png. Make it into a png!" in errstr)
        idx=0
        count=0
        while idx < len(errstr):
            idx = errstr.find("Dimensions",idx)
            if idx == -1:
                self.assertEqual(count,2)
                break
            else:
                count += 1
                idx += len("Dimensions")
        self.assertTrue("ERROR: The mask image's length and width do not seem to be the same as the base image's." in errstr)
        self.assertTrue("is not single-channel." in errstr)
        self.assertTrue("is not a png. Make it into a png!" in errstr)
        os.system('rm vs6.log')
        print("CASE 6 validated.")
        
        print("\nCASE 7: Validating behavior when at least one mask file is not present...") 
#        myval = DSD_Validator(validatorRoot + 'lorem_NC2016_UnitTest_Splice_ImgOnly_p-baseline_3/lorem_NC2016_UnitTest_Splice_ImgOnly_p-baseline_3.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv')
        myval = os.system("python2 validator.py -nc --ncid {} -vt DSD -s {} -x {} -p {} {}{}> vs7.log".format(NCID,validatorRoot + 'lorem_NC2016_UnitTest_Splice_ImgOnly_p-baseline_3/lorem_NC2016_UnitTest_Splice_ImgOnly_p-baseline_3.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv',procs,identify_string,nm_string))//256 
        errstr = msgcapture("vs7.log")
#        with stdout_redirect(StringIO.StringIO()) as errmsg:
#            result=myval.fullCheck(True,identify,NCID,neglectMask)
#        errmsg.seek(0)
#        errstr = errmsg.read()
#        errstr,val = print_capture('myval.fullCheck(True,identify,NCID,neglectMask)')
        self.assertEqual(myval,1)
        idx=0
        count=0
        while idx < len(errstr):
            idx = errstr.find("does not exist! Did you name it wrong?",idx)
            if idx == -1:
                self.assertEqual(count,3)
                break
            else:
                count += 1
                idx += len("does not exist! Did you name it wrong?")
        os.system('rm vs7.log')
        print("CASE 7 validated.")
        
        print("\nALL DSD VALIDATION TESTS SUCCESSFULLY PASSED.")
        
