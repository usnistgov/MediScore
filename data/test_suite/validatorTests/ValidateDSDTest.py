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

print("BASIC FUNCTIONALITY validation of DSDValidator.py beginning...")
myval = DSD_Validator(validatorRoot + 'lorem_NC2016_Splice_ImgOnly_p-baseline_1/lorem_NC2016_Splice_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv')
self.assertEqual(myval.fullCheck(),0)
print("BASIC FUNCTIONALITY validated.")

errmsg = ""
#Same checks as Validate SSD, but applied to different files
print("\nBeginning experiment ID naming error validations. Expect ERROR printouts for the next couple of cases. This is normal here.")
print("\nCASE 0: Validating behavior when files don't exist.") 
myval = DSD_Validator(validatorRoot + 'emptydir_NC2016_Splice_ImgOnly_p-baseline_1/emptydir_NC2016_Splice_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-splice-index0.csv')
with stdout_redirect(StringIO.StringIO()) as errmsg:
  result=myval.fullCheck()
errmsg.seek(0)
self.assertEqual(result,1)
self.assertTrue("ERROR: I can't find your system output" in errmsg.read())
self.assertTrue("ERROR: I can't find your index file" in errmsg.read())
print("CASE 0 validated.")

print("\nCASE 1: Validating behavior when detecting consecutive underscores ('_') in name...")
myval = DSD_Validator(validatorRoot + 'lorem__NC2016_Spl_ImgOnly_p-baseline_1/lorem__NC2016_Spl_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv')
with stdout_redirect(StringIO.StringIO()) as errmsg:
  result=myval.fullCheck()
errmsg.seek(0)
self.assertEqual(result,1)
self.assertTrue("ERROR: What kind of task is" in errmsg.read())
print("CASE 1 validated.")

print("\nCASE 2: Validating behavior when detecting excessive underscores elsewhere...")
myval = DSD_Validator(validatorRoot + 'lor_em_NC2016_Manipulation_ImgOnly_p-baseline_1/lor_em_NC2016_Manipulation_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv')
with stdout_redirect(StringIO.StringIO()) as errmsg:
  result=myval.fullCheck()
errmsg.seek(0)
self.assertEqual(result,1)
self.assertTrue("ERROR: What kind of task is" in errmsg.read())
print("CASE 2 validated.")

print("\nCASE 3: Validating behavior when detecting '+' in file name and an unrecogized task...\n")
myval = DSD_Validator(validatorRoot + 'lorem+_NC2016_Removal_ImgOnly_p-baseline_1/lorem+_NC2016_Removal_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv')
with stdout_redirect(StringIO.StringIO()) as errmsg:
  result=myval.fullCheck()
errmsg.seek(0)
self.assertEqual(result,1)
self.assertTrue("ERROR: The team name must not include characters" in errmsg.read())
self.assertTrue("ERROR: What kind of task is" in errmsg.read())
print("CASE 3 validated. Validating syntactic content of system output.")

print("\nCASE 4: Validating behavior for incorrect headers, duplicate rows, and different number of rows than in index file...")
myval = DSD_Validator(validatorRoot + 'lorem_NC2016_Splice_ImgOnly_p-baseline_2/lorem_NC2016_Splice_ImgOnly_p-baseline_2.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv')
with stdout_redirect(StringIO.StringIO()) as errmsg:
  result=myval.fullCheck()
errmsg.seek(0)
self.assertEqual(result,1)
self.assertTrue("ERROR: Your header(s)" in errmsg.read())
self.assertTrue("ERROR: Your system output contains duplicate rows" in errmsg.read())
self.assertTrue("ERROR: The number of rows in the system output does not match the number of rows in the index file." in errmsg.read())
print("CASE 4 validated.")

print("\nCase 5: Validating behavior when the number of columns in the system output is not equal to 5.")
myval = DSD_Validator(validatorRoot + 'lorem_NC2016_Splice_ImgOnly_p-baseline_4/lorem_NC2016_Splice_ImgOnly_p-baseline_4.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv')
with stdout_redirect(StringIO.StringIO()) as errmsg:
  result=myval.fullCheck()
errmsg.seek(0)
self.assertEqual(result,1)
self.assertTrue("ERROR: The number of columns of the system output file must be equal to 5. Are you using '|' to separate your columns?" in errmsg.read())
print("CASE 5 validated.")

print("\nCASE 6: Validating behavior for mask semantic deviations. NC2016-1893.jpg and NC2016_6847-mask.jpg are (marked as) jpg's. NC2016_1993-mask.png is not single-channel. NC2016_4281-mask.png doesn't have the same dimensions...")
myval = DSD_Validator(validatorRoot + 'ipsum_NC2016_Splice_ImgOnly_p-baseline_1/ipsum_NC2016_Splice_ImgOnly_p-baseline_1.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv')
with stdout_redirect(StringIO.StringIO()) as errmsg:
  result=myval.fullCheck()
errmsg.seek(0)
self.assertEqual(result,1)
self.assertTrue("is not a png. Make it into a png!" in errmsg.read())
idx=0
count=0
while idx < len(errmsg.read()):
  idx = errmsg.read().find("Dimensions",idx)
  if idx == -1:
    self.assertEqual(count,2)
    break
  else:
    count += 1
    idx += len("Dimensions")
self.assertTrue("ERROR: The mask image's length and width do not seem to be the same as the base image's." in errmsg.read())
self.assertTrue("is not a png. Make it into a png!" in errmsg.read())
print("CASE 6 validated.")

print("\nCASE 7: Validating behavior when mask file is not present...") 
myval = DSD_Validator(validatorRoot + 'lorem_NC2016_Splice_ImgOnly_p-baseline_3/lorem_NC2016_Splice_ImgOnly_p-baseline_3.csv',validatorRoot + 'NC2016_Test0516_dfz/indexes/NC2016-splice-index.csv')
with stdout_redirect(StringIO.StringIO()) as errmsg:
  result=myval.fullCheck()
errmsg.seek(0)
self.assertEqual(result,1)
idx=0
count=0
while idx < len(errmsg.read()):
  idx = errmsg.read().find("does not exist! Did you name it wrong?",idx)
  if idx == -1:
    self.assertEqual(count,2)
    break
  else:
    count += 1
    idx += len("does not exist! Did you name it wrong?")
print("CASE 7 validated.")

print("\nALL DSD VALIDATION TESTS SUCCESSFULLY PASSED.")

