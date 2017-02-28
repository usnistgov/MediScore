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

        print("BASIC FUNCTIONALITY validation of SSDValidator.r beginning...")
        myval = SSD_Validator(validatorRoot + 'foo_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_1/foo_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv')
        self.assertEqual(myval.fullCheck(True,True),0)
        print("BASIC FUNCTIONALITY validated.")
        
        print("\nBeginning experiment ID naming error validations. Expect ERROR printouts for the next couple of cases. This is normal here.")
        print("CASE 0: Validating behavior when files don't exist.")
        
        myval = SSD_Validator(validatorRoot + 'emptydir_NC2016_UnitTest_Splice_ImgOnly_p-baseline_1/foo__NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-manipulation-index0.csv')
        
        with stdout_redirect(StringIO.StringIO()) as errmsg:
            result=myval.fullCheck(True,True) 
            
        errmsg.seek(0)
        self.assertEqual(result,1)
        errstr = errmsg.read() #NOTE: len(errmsg.read())==0, but when you set it equal, you get the entire string. What gives?
        self.assertTrue("ERROR: I can't find your system output" in errstr)
        errmsg.close()
        
        print("CASE 0 validated.")
        
        print("\nCASE 1: Validating behavior when detecting consecutive underscores ('_') in name...")
        myval = SSD_Validator(validatorRoot + 'foo__NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_1/foo__NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv')
        
        with stdout_redirect(StringIO.StringIO()) as errmsg:
            result=myval.fullCheck(True,True)
        errmsg.seek(0)
        self.assertEqual(result,1)
        errstr = errmsg.read()
        self.assertTrue("ERROR: What kind of task is" in errstr)
        print("CASE 1 validated.")
        
        print("\nCASE 2: Validating behavior when detecting excessive underscores elsewhere...")
        myval = SSD_Validator(validatorRoot + 'fo_o_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_1/fo_o_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv')
        
        with stdout_redirect(StringIO.StringIO()) as errmsg:
            result=myval.fullCheck(True,True)
        errmsg.seek(0)
        self.assertEqual(result,1)
        errstr = errmsg.read()
        self.assertTrue("ERROR: What kind of task is" in errstr)
        print("CASE 2 validated.")
        
        print("\nCASE 3: Validating behavior when detecting '+' in file name and an unrecognized task...")
        myval = SSD_Validator(validatorRoot + 'foo+_NC2016_UnitTest_Manip_ImgOnly_p-baseline_1/foo+_NC2016_UnitTest_Manip_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv')
        
        with stdout_redirect(StringIO.StringIO()) as errmsg:
            result=myval.fullCheck(True,True)
        errmsg.seek(0)
        self.assertEqual(result,1)
        errstr = errmsg.read()
        self.assertTrue("ERROR: The team name must not include characters" in errstr)
        self.assertTrue("ERROR: What kind of task is" in errstr)
        print("CASE 3 validated.")
 
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
        print("Validating syntactic content of system output.\nCASE 4: Validating behavior for incorrect headers, duplicate rows, and different number of rows than in index file...")
        print("CASE 4a: Validating behavior for incorrect headers")
        myval = SSD_Validator(validatorRoot + 'foo_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_2/foo_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_2.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv') 
        with stdout_redirect(StringIO.StringIO()) as errmsg:
            result=myval.fullCheck(True,True)
        errmsg.seek(0)
        self.assertEqual(result,1)
        errstr = errmsg.read()
        self.assertTrue("ERROR: The required column" in errstr)

        print("CASE 4b: Validating behavior for duplicate rows and different number of rows than in index file...")
        myval = SSD_Validator(validatorRoot + 'foob_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_2/foob_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_2.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv')
        with stdout_redirect(StringIO.StringIO()) as errmsg:
            result=myval.fullCheck(True,True)
        errmsg.seek(0)
        self.assertEqual(result,1)
        errstr = errmsg.read()
        self.assertTrue("ERROR: Your system output contains duplicate rows" in errstr)
        self.assertTrue("ERROR: The number of rows in your system output does not match the number of rows in the index file." in errstr)
        print("CASE 4 validated.")
        
        print("\nCASE 5: Validating behavior when mask is not a png...")
        myval = SSD_Validator(validatorRoot + 'bar_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_1/bar_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv')
        
        with stdout_redirect(StringIO.StringIO()) as errmsg:
            result=myval.fullCheck(True,True)
        errmsg.seek(0)
        self.assertEqual(result,1)
        errstr = errmsg.read()
        self.assertTrue("is not a png. Make it into a png!" in errstr)
        print("CASE 5 validated.")
        
        print("\nCASE 6: Validating behavior when mask is not single channel and when mask does not have the same dimensions.")
        myval = SSD_Validator(validatorRoot + 'baz_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_1/baz_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv')
        
        with stdout_redirect(StringIO.StringIO()) as errmsg:
            result=myval.fullCheck(True,True)
        errmsg.seek(0)
        self.assertEqual(result,1)
        errstr = errmsg.read()
        self.assertEqual(errstr.count("Dimensions"),2)
        self.assertTrue("ERROR: The mask image's length and width do not seem to be the same as the base image's." in errstr)
        self.assertTrue("is not single-channel. Make it single-channel." in errstr)
        print("CASE 6 validated.")
        
        print("\nCASE 7: Validating behavior when system output column number is less than 3.") 
        myval = SSD_Validator(validatorRoot + 'foo_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_3/foo_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_3.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv')
        
        with stdout_redirect(StringIO.StringIO()) as errmsg:
            result=myval.fullCheck(True,True)
        errmsg.seek(0)
        self.assertEqual(result,1)
        errstr = errmsg.read()
        self.assertTrue("ERROR: The number of columns of the system output file must be at least 2. Are you using '|' to separate your columns?" in errstr)
        print("CASE 7 validated.")
        
        print("\nCASE 8: Validating behavior when mask file is not present.") 
        myval = SSD_Validator(validatorRoot + 'foo_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_4/foo_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_4.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv')
        
        with stdout_redirect(StringIO.StringIO()) as errmsg:
            result=myval.fullCheck(True,True)
        errmsg.seek(0)
        self.assertEqual(result,1)
        errstr = errmsg.read()
        self.assertTrue("does not exist! Did you name it wrong?" in errstr)
        
        print("CASE 8 validated.")
        
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
        myval = DSD_Validator(validatorRoot + 'lorem_NC2016_UnitTest_Splice_ImgOnly_p-baseline_1/lorem_NC2016_UnitTest_Splice_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv')
        self.assertEqual(myval.fullCheck(True,True),0)
        print("BASIC FUNCTIONALITY validated.")
        
        errmsg = ""
        #Same checks as Validate SSD, but applied to different files
        print("\nBeginning experiment ID naming error validations. Expect ERROR printouts for the next couple of cases. This is normal here.")
        print("\nCASE 0: Validating behavior when files don't exist.") 
        myval = DSD_Validator(validatorRoot + 'emptydir_NC2016_UnitTest_Splice_ImgOnly_p-baseline_1/emptydir_NC2016_UnitTest_Splice_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-splice-index0.csv')
        with stdout_redirect(StringIO.StringIO()) as errmsg:
            result=myval.fullCheck(True,True)
        errmsg.seek(0)
        self.assertEqual(result,1)
        errstr = errmsg.read()
        self.assertTrue("ERROR: I can't find your system output" in errstr)
        self.assertTrue("ERROR: I can't find your index file" in errstr)
        print("CASE 0 validated.")
        
        print("\nCASE 1: Validating behavior when detecting consecutive underscores ('_') in name...")
        myval = DSD_Validator(validatorRoot + 'lorem__NC2016_UnitTest_Spl_ImgOnly_p-baseline_1/lorem__NC2016_UnitTest_Spl_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv')
        with stdout_redirect(StringIO.StringIO()) as errmsg:
            result=myval.fullCheck(True,True)
        errmsg.seek(0)
        self.assertEqual(result,1)
        errstr = errmsg.read()
        self.assertTrue("ERROR: What kind of task is" in errstr)
        print("CASE 1 validated.")
        
        print("\nCASE 2: Validating behavior when detecting excessive underscores elsewhere...")
        myval = DSD_Validator(validatorRoot + 'lor_em_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_1/lor_em_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv')
        with stdout_redirect(StringIO.StringIO()) as errmsg:
            result=myval.fullCheck(True,True)
        errmsg.seek(0)
        self.assertEqual(result,1)
        errstr = errmsg.read()
        self.assertTrue("ERROR: What kind of task is" in errstr)
        print("CASE 2 validated.")
        
        print("\nCASE 3: Validating behavior when detecting '+' in file name and an unrecogized task...\n")
        myval = DSD_Validator(validatorRoot + 'lorem+_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_1/lorem+_NC2016_UnitTest_Manipulation_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv')
        with stdout_redirect(StringIO.StringIO()) as errmsg:
            result=myval.fullCheck(True,True)
        errmsg.seek(0)
        self.assertEqual(result,1)
        errstr = errmsg.read()
        self.assertTrue("ERROR: The team name must not include characters" in errstr)
        self.assertTrue("ERROR: What kind of task is" in errstr)
        print("CASE 3 validated.")
     
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
        myval = DSD_Validator(validatorRoot + 'lorem_NC2016_UnitTest_Splice_ImgOnly_p-baseline_2/lorem_NC2016_UnitTest_Splice_ImgOnly_p-baseline_2.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv')
        with stdout_redirect(StringIO.StringIO()) as errmsg:
            result=myval.fullCheck(True,True)
        errmsg.seek(0)
        self.assertEqual(result,1)
        errstr = errmsg.read()
        self.assertTrue("ERROR: The required column" in errstr)
        print("CASE 4b: Validating behavior for duplicate rows and different number of rows than in index file...")
        myval = DSD_Validator(validatorRoot + 'loremb_NC2016_UnitTest_Splice_ImgOnly_p-baseline_2/loremb_NC2016_UnitTest_Splice_ImgOnly_p-baseline_2.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv')
        with stdout_redirect(StringIO.StringIO()) as errmsg:
            result=myval.fullCheck(True,True)
        errmsg.seek(0)
        self.assertEqual(result,1)
        errstr = errmsg.read()
#        self.assertTrue("ERROR: Row" in errstr) #TODO: temporary measure until we get duplicates back
        self.assertTrue("ERROR: The number of rows in your system output does not match the number of rows in the index file." in errstr)
        print("CASE 4 validated.")
        
#        print("\nCase 5: Validating behavior when the number of columns in the system output is less than 5.")
#        myval = DSD_Validator(validatorRoot + 'lorem_NC2016_UnitTest_Splice_ImgOnly_p-baseline_4/lorem_NC2016_UnitTest_Splice_ImgOnly_p-baseline_4.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv')
#        with stdout_redirect(StringIO.StringIO()) as errmsg:
#            result=myval.fullCheck(True)
#        errmsg.seek(0)
#        self.assertEqual(result,1)
#        errstr = errmsg.read()
#        self.assertTrue("ERROR: The number of columns of the system output file must be at least 5. Are you using '|' to separate your columns?" in errstr)
#        print("CASE 5 validated.")
        
        print("\nCASE 6: Validating behavior for mask semantic deviations. NC2016-1893.jpg and NC2016_6847-mask.jpg are (marked as) jpg's. NC2016_1993-mask.png is not single-channel. NC2016_4281-mask.png doesn't have the same dimensions...")
        myval = DSD_Validator(validatorRoot + 'ipsum_NC2016_UnitTest_Splice_ImgOnly_p-baseline_1/ipsum_NC2016_UnitTest_Splice_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv')
        with stdout_redirect(StringIO.StringIO()) as errmsg:
            result=myval.fullCheck(True,True)
        errmsg.seek(0)
        self.assertEqual(result,1)
        errstr = errmsg.read()
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
        self.assertTrue("is not single-channel. Make it single-channel." in errstr)
        self.assertTrue("is not a png. Make it into a png!" in errstr)
        print("CASE 6 validated.")
        
        print("\nCASE 7: Validating behavior when at least one mask file is not present...") 
        myval = DSD_Validator(validatorRoot + 'lorem_NC2016_UnitTest_Splice_ImgOnly_p-baseline_3/lorem_NC2016_UnitTest_Splice_ImgOnly_p-baseline_3.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv')
        with stdout_redirect(StringIO.StringIO()) as errmsg:
            result=myval.fullCheck(True,True)
        errmsg.seek(0)
        self.assertEqual(result,1)
        errstr = errmsg.read()
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
        print("CASE 7 validated.")
        
        print("\nALL DSD VALIDATION TESTS SUCCESSFULLY PASSED.")
        
