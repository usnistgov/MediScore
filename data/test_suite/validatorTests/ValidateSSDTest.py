import sys,contextlib,StringIO,unittest

#EDIT: to be joined with validator object

@contextlib.contextmanager
def stdout_redirect(where):
  sys.stdout = where
  try:
    yield where
  finally:
    sys.stdout = sys.__stdout__

validatorRoot = '../data/test_suite/validatorTests/'
quiet = 1

print("BASIC FUNCTIONALITY validation of SSDValidator.r beginning...")
myval = validator(validatorRoot + 'foo_NC2016_Manipulation_ImgOnly_p-baseline_1/foo_NC2016_Manipulation_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv')
self.assertEqual(myval.fullCheck(),0)
print("BASIC FUNCTIONALITY validated.")

print("\nBeginning experiment ID naming error validations. Expect ERROR printouts for the next couple of cases. This is normal here.")
print("CASE 0: Validating behavior when files don't exist.")

myval = validator(validatorRoot + 'emptydir_NC2016_Splice_ImgOnly_p-baseline_1/foo__NC2016_Manipulation_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-manipulation-index0.csv')

with stdout_redirect(StringIO.StringIO()) as errmsg:
  result=myval.fullCheck() 
  
errmsg.seek(0)
self.assertEqual(result,1)
self.assertTrue("ERROR: I can't find your system output" in errmsg.read())
errmsg.close()

print("CASE 0 validated.")

print("\nCASE 1: Validating behavior when detecting consecutive underscores ('_') in name...")
myval = validator(validatorRoot + 'foo__NC2016_Manipulation_ImgOnly_p-baseline_1/foo__NC2016_Manipulation_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv')

with stdout_redirect(StringIO.StringIO()) as errmsg:
  result=myval.fullCheck()
errmsg.seek(0)
self.assertEqual(result,1)
self.assertTrue("ERROR: What kind of task is" in errmsg.read())
print("CASE 1 validated.")

print("\nCASE 2: Validating behavior when detecting excessive underscores elsewhere...")
myval = validator(validatorRoot + 'fo_o_NC2016_Manipulation_ImgOnly_p-baseline_1/fo_o_NC2016_Manipulation_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv')

with stdout_redirect(StringIO.StringIO()) as errmsg:
  result=myval.fullCheck()
errmsg.seek(0)
self.assertEqual(result,1)
self.assertTrue("ERROR: What kind of task is" in errmsg.read())
print("CASE 2 validated.")

print("\nCASE 3: Validating behavior when detecting '+' in file name and an unrecognized task...\n")
myval = validator(validatorRoot + 'foo+_NC2016_Manipulation_ImgOnly_p-baseline_1/foo+_NC2016_Manipulation_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv')

with stdout_redirect(StringIO.StringIO()) as errmsg:
  result=myval.fullCheck()
errmsg.seek(0)
self.assertEqual(result,1)
self.assertTrue("ERROR: The team name must not include characters" in errmsg.read())
self.assertTrue("ERROR: What kind of task is" in errmsg.read())
print("CASE 3 validated. Validating syntactic content of system output.")

print("\nCASE 4: Validating behavior for incorrect headers, duplicate rows, and different number of rows than in index file...")
myval = validator(validatorRoot + 'foo_NC2016_Manipulation_ImgOnly_p-baseline_2/foo_NC2016_Manipulation_ImgOnly_p-baseline_2.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv')

with stdout_redirect(StringIO.StringIO()) as errmsg:
  result=myval.fullCheck()
errmsg.seek(0)
self.assertEqual(result,1)
self.assertTrue("ERROR: Your header(s)" in errmsg.read())
self.assertTrue("ERROR: Your system output contains duplicate rows" in errmsg.read())
self.assertTrue("ERROR: The number of rows in the system output does not match the number of rows in the index file." in errmsg.read())
print("CASE 4 validated.")

print("\nCASE 5: Validating behavior when mask is not a png...")
myval = validator(validatorRoot + 'bar_NC2016_Manipulation_ImgOnly_p-baseline_1/bar_NC2016_Manipulation_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv')

with stdout_redirect(StringIO.StringIO()) as errmsg:
  result=myval.fullCheck()
errmsg.seek(0)
self.assertEqual(result,1)
self.assertTrue("is not a png. Make it into a png!" in errmsg.read())
print("CASE 5 validated.")

print("\nCASE 6: Validating behavior when mask is not single channel and when mask does not have the same dimensions.")
myval = validator(validatorRoot + 'baz_NC2016_Manipulation_ImgOnly_p-baseline_1/baz_NC2016_Manipulation_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv')

with stdout_redirect(StringIO.StringIO()) as errmsg:
  result=myval.fullCheck()
errmsg.seek(0)
self.assertEqual(result,1)
self.assertEqual(errmsg.read().count("Dimensions"),2)
self.assertTrue("ERROR: The mask image's length and width do not seem to be the same as the base image's." in errmsg.read())
print("CASE 6 validated.")

print("\nCASE 7: Validating behavior when system output column number is not equal to 3.") 
myval = validator(validatorRoot + 'foo_NC2016_Manipulation_ImgOnly_p-baseline_3/foo_NC2016_Manipulation_ImgOnly_p-baseline_3.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv')

with stdout_redirect(StringIO.StringIO()) as errmsg:
  result=myval.fullCheck()
errmsg.seek(0)
self.assertEqual(result,1)
self.assertTrue("ERROR: Your header(s)" in errmsg.read())
self.assertTrue("ERROR: The number of columns of the system output file must be equal to 3. Are you using '|' to separate your columns?" in errmsg.read())
print("CASE 7 validated.")

print("\nCASE 8: Validating behavior when mask file is not present.") 
myval = validator(validatorRoot + 'foo_NC2016_Manipulation_ImgOnly_p-baseline_4/foo_NC2016_Manipulation_ImgOnly_p-baseline_4.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-manipulation-index.csv')

with stdout_redirect(StringIO.StringIO()) as errmsg:
  result=myval.fullCheck()
errmsg.seek(0)
self.assertEqual(result,1)
self.assertTrue("does not exist! Did you name it wrong?" in errmsg.read())

print("CASE 8 validated.\n")

print("\nALL SSD VALIDATION TESTS SUCCESSFULLY PASSED.")
