"""
* File: CrossTeamLocalizationReport.py
* Date Started: 4/26/2017
* Date Updated: 5/19/2017
* Status: Complete
	- Fix filepath for images before inserting into database
	- Add from specified directory
	- Sort probes based on max MCC per probe
	- Sort teams based on average MCC per team
	- Option to specify table name
	- preQuery option added
* Status: Development
	- NA
* Status: Future
	- Mouse-over image pop-up
	- Possibly autogenerate thumbnails to improve web page loading time
		* PHP + Javascript?

* Description: This code contains functions for generating cross team localization reports

* Requirements: This code requires the following packages:
	
	-

  The rest are available on your system.

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
import argparse
import csv
import sqlite3
import re
import os
import fnmatch
from string import Template


def main():

	##### Command line arguments #####
	parser = argparse.ArgumentParser(description="Creates a cross-team localization report.")

	# If scoreFilePath given, must associate expName with it
	parser.add_argument('--scoreFilePath', type=str, default='None', nargs='+',
		help='Filepath to CSV containing score results: [e.g. /Users/alp3/DryRunScores/HW_NC17_DRYRUN17_Manipulation_ImgOnly_p-baseline_1-mask_scores_perimage.csv]. If none given, will query already existing data in the database.')

	parser.add_argument('--expName', type=str, default='None', nargs='+',
		help='Experiment Name associated with respective scoreFilePath: [e.g. HW_p-baseline_1]')

	parser.add_argument('--delimiter', '-d', type=str, default='|',
		help='Use to specify if your data files are not delimited by | (pipes). [e.g. ,]')

	parser.add_argument('--addOnly', '-a', action='store_true',
		help='Use this option to add CSV files to database without producing an HTML file')

	parser.add_argument('--addDir', type=str, default=['None', 'Manipulation*perimage.csv'], nargs=2,
		help='Use this to add all CSVs in a specified directory with glob pattern [e.g. /Users/alp3/DryRunScores/ *Manipulation*perimage.csv]. This will automatically generate experiment names in the form of <TEAM>_<SYS>_<VERSION> (e.g. HW_p-baseline_1')

	parser.add_argument('--fixFilePath', '-fp', type=str, default=['None','None'], nargs=2,
		help='Use this option to correct filepaths of images in CSV file before data is inserted into database [e.g. /oldDirectory/ /newDirectory/]')

	parser.add_argument('--dbName', '-db', type=str, default='ScoresPerImage.db',
		help='Name of database to be created and/or queried: [e.g. DryRunScores.db]. Defaults to ScoresPerImage.db')

	parser.add_argument('--tableName', '-tn', type=str, default='CombinedCSVData',
		help='Name of table to be created and/or queried: [e.g. MaskScoresPerImage]. Defaults to CombinedCSVData')

	parser.add_argument('--preQuery', '-pq', type=str, default='None',
		help="For Jon")

	parser.add_argument('--queryProbe', '-qp', type=str, default='all', nargs='+',
		help='ProbeIDs to be displayed in HTML file: [e.g. 003eaa9f0f222263550b95e0ab994f33]. If not used, defaults to creating a report with all probes.')

	parser.add_argument('--queryExp', '-qe', type=str, default='all', nargs='+',
		help='Experiments to be displayed in HTML file: [e.g. HW_p-baseline_1]. Default is for all experiments. Experiment names must exist in database already and name must match exactly, or be added with --scoreFilePath and --expName.')

	parser.add_argument('--queryMCC', '-qm', type=str, default='all', nargs='+',
		help='Use to query MCC, numbers must be between. [-1, 1]: [e.g. "> 0.5"]. Multiple arguments will be ANDed together. [e.g. "> 0" "< 0.5" is equivalent to MCC > 0 AND MCC < 0.5] Default will query all MCCs of all values.')

	parser.add_argument('--sortProbes', '-sp', type=str, default='None',
		help='Use to sort output results by probes based on descending (desc) or ascending (asc) order of max MCC for each probe [e.g. desc]. Default is none.')

	parser.add_argument('--sortTeams', '-st', type=str, default='None',
		help='Use to sort output results by on descending (desc) or ascending (asc) order average MCC scores of each time [e.g desc]. Default is none.')

	parser.add_argument('--outputFileName', '-out', type=str, default='output.html',
		help='Name of output file: [e.g. AllDryRunResults.html]. Defaults to output.html')

	parser.add_argument('--verbose', '-v', action='store_true',
		help='Control print output.')

	args = parser.parse_args()

	if (len(args.scoreFilePath) != len(args.expName)):
		parser.error('--scoreFilePath requires an argument with the --expName option for each argument provided in --scoreFilePath.')

	if args.sortProbes not in ['None', 'asc', 'desc']:
		parser.error('--sortProbes argument must be asc or desc')

	if args.sortTeams not in ['None', 'asc', 'desc']:
		parser.error('--sortTeams argument must be asc or desc')

	tableName = args.tableName

	# Inserts data from each CSV file and associates it with respective experiment name
	if (args.scoreFilePath > 0 and args.scoreFilePath != 'None'):
		for dataFilePath, expName in zip(args.scoreFilePath, args.expName):
			if args.verbose:
				print(addToDB(dataFilePath, args.dbName, tableName, expName, args.delimiter, args.fixFilePath))
			else:
				addToDB(dataFilePath, args.dbName, tableName, expName, args.delimiter, args.fixFilePath)

	# If addDir option used, searches directory and adds all Manipulation perimage CSV files to specified database
	if args.addDir[0] != 'None':
		if args.verbose:
			print(addFromDir(args.addDir, args.dbName, tableName, args.delimiter, args.fixFilePath, args.verbose))
		else:
			addFromDir(args.addDir, args.dbName, tableName, args.delimiter, args.fixFilePath, args.verbose)

	if not args.addOnly:
		
		if args.preQuery != 'None':
			print(createTable(args.dbName, args.preQuery))

		generalInfo, perExpInfo = queryDB(args.dbName, tableName, args.queryProbe, args.queryExp, args.queryMCC, args.verbose)
		
		
		# Output overallResults to an HTML file
		if args.verbose:
			print(htmlReport(generalInfo, perExpInfo, args.outputFileName, args.sortProbes, args.sortTeams))
		else:
			htmlReport(generalInfo, perExpInfo, args.outputFileName, args.sortProbes, args.sortTeams)



def createTable(db, query):
	# Creates a table to in the database that will be used for HTML output

	conn = sqlite3.connect(db)
	c = conn.cursor()

	sqlText = query
	c.execute(sqlText)

	conn.commit()
	conn.close()

	return 'Table created using %s' % query


def addFromDir(scoresDir,  dbName, tableName, delimiter, fixFilePath, verbose):
	# Adds all CSV files matching *Manipulation*perimage.csv or specified pattern in a given directory into specified database

	for root, dirs, files in os.walk(scoresDir[0]):
		for file in files:
			if fnmatch.fnmatch(file, scoresDir[1]):

				# Get exp name
				temp = file.split('_')
				version = temp[-3].split('-')[0]
				expName = '_'.join([temp[0], temp[5], version])


				if verbose: 
			 		print(addToDB(os.path.join(root, file), dbName, tableName, expName, delimiter, fixFilePath))
				else:
				 	addToDB(os.path.join(root, file), dbName, tableName, expName, delimiter, fixFilePath)

	return 'Files from %s sucessfully added' % scoresDir

def addToDB(CSVFilePath, db, tableName, expName, delimiter, fixFilePath):
	# Adds CSV data to new or existing database (db) 

	oldDir = fixFilePath[0]
	newDir = fixFilePath[1]

	conn = sqlite3.connect(db)
	conn.text_factory = str
	c = conn.cursor()

	with open(CSVFilePath, "rb") as f:
		reader = csv.reader(f, delimiter=delimiter)

		header = True
		
		try:
			for row in reader:
			
				# Need to get column names from first row of csv
				if header:
					header = False	

					sqlText = "CREATE TABLE IF NOT EXISTS %s(%s)" % (tableName, ', '.join(['ExpName TEXT', ', '.join(['%s TEXT' % column for column in row])]))
					c.execute(sqlText)			

					# Check to see if EXPID already exists in table before adding additional rows in table
					sqlText = "SELECT DISTINCT ExpName from %s" % tableName
					c.execute(sqlText)
					results = c.fetchall()
					for element in results:
						if element[0] == expName:
							return 'Experiment matching Experiment Name %s already in database.' % expName


					sqlText = "INSERT INTO %s VALUES(%s)" % (tableName, ', '.join(['?', ', '.join(['?' for column in row])]))
					
				else:
					row.insert(0, expName)
					insertRow = row[:]

					# Updates filepath of images before inserting data into database
					if oldDir != 'None':
					 	insertRow = []
					 	for column in row:
					 		insertRow.append(re.sub(oldDir, newDir, column))
					c.execute(sqlText, insertRow)
		
		except csv.Error, e:
			return 'File %s was unable to be loaded into the DB' % CSVFilePath

	conn.commit()
	conn.close()

	return '%s successfully added' % expName


def queryDB(database, tableName, probes, exps, MCC, verbose):
	# Query database based on user input: 
	#	If no input, queries all probes
	#	If only probes specified, queries all experiments for specified probes
	#	If only experiements specified, queries all probes for specified experiments
	#	If probes and experiments specified, queries specified probes for specified experiments
	#   If probes, experiments, and MCC specified, queries based on those parameters
	
	conn = sqlite3.connect(database)
	c = conn.cursor()

	# List of elements that I want to query that are the same among all experiements
	# Can change this later to get these from command line?
	queryElements = ['ProbeFileID', 'AggMaskFileName', 'ProbeMaskFileName']

	# List of elements that are unique to each experiment
	# Using cast(MCC as real) instead of just MCC to handle the case where MCC gets stored in scientific notation (e-05)
	# Can change this later to get these from command line?
	perExpElements = ['ExpName', 'cast(MCC as real)', 'Scored', 'ColMaskFileName']

	queryResults =[]
	perExpResults = []

	# Constructs probesList 
	if 'all' in probes:
		if verbose:
			print("Query all probes")

		sqlText = "SELECT DISTINCT ProbeFileID FROM %s" % tableName
		c.execute(sqlText)
		probesResults = c.fetchall()

		probesList = []
		for element in probesResults:
			probesList.append(element[0])

	else:
		if verbose:
			print("Query the following probes: %s" % ', '.join(probes))
		probesList = probes

	# Constructs expsList
	if 'all' in exps:
		if verbose:
			print("Query all experiments")

		sqlText = "SELECT DISTINCT ExpName FROM %s" % tableName
		c.execute(sqlText)
		expsResults = c.fetchall()

		expsList = []
		for element in expsResults:
			expsList.append(element[0])

	else:
		if verbose:	
			print("Query the following experiments: %s" % ', '.join(exps))
		expsList = exps

	if 'all' in MCC:
		if verbose:
			print("Query all MCCs")
		mccQuery = ''
	elif len(MCC) == 1:
		if verbose:
			print("Query MCCs %s" % MCC[0])
		mccQuery = ' AND cast(MCC as real) ' + MCC[0]
	else:
		if verbose:
			print("Query MCCs %s" % ' and '.join(MCC))
		mccQuery = ' AND cast(MCC as real) ' + ' AND cast(MCC as real) '.join(MCC)		

	count = 0
	total = len(probesList)


	#### Maybe use the following: WHERE ProbeFileID IN ('probe1', 'probe2', ...)
	#### Then, can avoid looping through probes and exps individually

	for probe in probesList:

		# Progress output
		if verbose:
			count += 1
			percentage = float(count) / total * 100
			msgText = "Querying is %.2f%% complete" % percentage
			print msgText, "               \r",

		# sqlText1 = "SELECT DISTINCT %s FROM %s WHERE ProbeFileID = '%s'" % (', '.join(queryElements), tableName, probe)
		# c.execute(sqlText1)
		# queryResults.append(c.fetchall())

		expsPerProbeList = []
		genPerProbeList = []
		for exp in expsList:
			sqlText1 = "SELECT %s FROM %s WHERE ProbeFileID = '%s' AND ExpName = '%s'%s" % (', '.join(queryElements), tableName, probe, exp, mccQuery)
			c.execute(sqlText1)
			genPerProbeList.append(c.fetchone())

			sqlText2 = "SELECT %s FROM %s WHERE ProbeFileID = '%s' AND ExpName = '%s'%s" % (', '.join(perExpElements), tableName, probe, exp, mccQuery)
			c.execute(sqlText2)
			results = c.fetchone()

			sqlText3 = "SELECT Scored FROM %s WHERE ProbeFileID = '%s' AND ExpName = '%s'" % (tableName, probe, exp)
			c.execute(sqlText3)
			isScored = c.fetchone()

			# Exits program if no data is found for a given probe and/or experiment name	
			if results == None and isScored == None:
				# print("Check input. Data for probe %s and experiment %s not found." % (probe, exp))
				# sys.exit()
				results = (exp, None, 'Not in Table', None)

			# Need to construct output if none exists for HTML ouput
			if results == None and isScored[0] == 'Y':
				results = (exp, 'NA', isScored[0], 'NA')
			elif results == None and isScored[0] == 'N':
				results = (exp, None, isScored[0], None)
			else:
				pass

			expsPerProbeList.append(results)
			
		perExpResults.append(expsPerProbeList)
		queryResults.append(genPerProbeList)

	if verbose:
		msgText = "Querying is 100.00% complete"
		print msgText, "               "

	conn.close()
	return queryResults, perExpResults

def htmlReport(general, perExp, outputFile, sortProbes, sortTeams):
	# Generates an HTML file with data from query to DB
	
	headers = ['Composite', 'Binarized']

	with open (outputFile, 'w') as output:

		#### Might be better to get maxMCC and avgMCC while querying the database instead of calculating here ####

		# If sortProbes option is used, sorts probes in order of max MCC for each probe
		if (sortProbes != 'None'):
			for probe in perExp:

				maxVal = max([sublist[1] for sublist in probe])
				probe.insert(0, maxVal)

			if sortProbes == 'desc':	
				perExp, general = zip(*sorted(zip(perExp, general), reverse=True))
			else:
				perExp, general = zip(*sorted(zip(perExp, general)))	

			for probe in perExp:
				del probe[0]

		# If sortTeams option is used, sorts teams in order of average MCC for each team
		if (sortTeams != 'None'):
			numExps = len(perExp[0])
			unpackExps = []
			for i in range(numExps):
				unpackExps.append([])

			for probe in perExp:
				for i in range(numExps):
					unpackExps[i].append(probe[i])

			# What happens when all values for MCC are NA?
			# Average is not correct if some probes for a team are not scored, but still get counted as having a score
			for exp in unpackExps:
				total = 0
				count = 0
				for probe in exp:
					total += probe[1]

				exp.insert(0, float(total)/len(exp))

			if sortTeams == 'desc':
				sortedExpList = sorted(unpackExps, reverse=True)
			else:
				sortedExpList = sorted(unpackExps)

			for exp in sortedExpList:
				del exp[0]

			perExp = []
			numProbes = len(sortedExpList[0])
			for i in range(numProbes):
				perExp.append([])

			for exp in sortedExpList:
				for i in range(numProbes):
					perExp[i].append(exp[i])	

		# Iterates through each probe and for each probe through each exp to generate report output
		resultsList = []
		for probeInfo, exps in zip(general, perExp):
			expResultsCombined = []
			for experiment in exps:
				
				if experiment[0] not in headers:
					headers.append(experiment[0])
				
				# If Scored is 'N' is an empty string, exp was not score for this particular probe
				if experiment[2] == 'Not in Table':
					experimentResults = '<td>Experiment/Probe not in table</td>'

				elif experiment[2] == 'N':
					experimentResults = '<td>Experiment was not evaluated for this probe</td>'
 
				elif experiment[2] == 'Y' and experiment[1] == 'NA':
					experimentResults = '<td>Experiment didn\'t meet query criteria</td>'

				else:
					experimentResults = '<td>MCC: ' + str(experiment[1]) + '<br><a href="' + experiment[3] + '" target="_blank"><img src="' + experiment[3] + '" alt="Evaluation Results" style="width:304px;height:228px;"></td>'

				expResultsCombined.append(experimentResults)
			
			expsJoined = '\n\t\t'.join(expResultsCombined)

			# In input CVS, there is no column that indicates location of this file
			# This code takes the filepath of another file in the same directory
			# And converts <ProbeMaskFileName>.ccm.png to <ProbeMaskFileName>.ccm-bin.png
			# Then, puts it together to get the correct directory and filename of the binarized reference mask
			for result in probeInfo:
				if result != None:

					binRefMaskFilePathList = result[1].split('/')[:-1]
					correctBinRefMaskFileName = result[2].split('/')[-1].split('.')[0] + '.ccm-bin.png'
					binRefMaskFilePathList.append(correctBinRefMaskFileName)

					# Testing with templates within this script:
					result = """<tr>
						<td>%s<br><a href="%s" target="_blank"><img src="%s" alt="Compositie Mask with Color" style="width:304px;height:228px;"></td>
 						<td><br><a href="%s" target="_blank"><img src="%s" alt="Binarized Reference Mask" style="width:304px;height:228px;"></td>
 						%s
					</tr>""" % (result[0], result[1], result[1], '/'.join(binRefMaskFilePathList), '/'.join(binRefMaskFilePathList), expsJoined)

					resultsList.append(result)
					break

		# Puts the pieces of HTML together and formats HTML based on 'outer' template
		headersJoined = '<th>' + '</th>\n\t\t<th>'.join(headers) + '</th>'
		rowsJoined = ''.join(resultsList)

		htmlOut = """<!DOCTYPE html>
			<html>
			<head>
			<style>
			table, th, td {
				border: 1px solid black;
				min-width: 304px;
				}
			</style>
			</head>
			<body>

			<table>
				<tr>
				%s
				</tr>	
			%s
			
			</table>
			</body>
			</html>
			""" % (headersJoined, rowsJoined)
		
		# Writes to output HTML file
		output.write(htmlOut)

	return "%s has been generated" % outputFile 

if __name__ == '__main__':
	main()