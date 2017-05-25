"""
* File: CrossTeamLocalizationReport.py
* Date Started: 4/26/2017
* Date Updated: 5/25/2017
* Status: Complete

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


def main():

	##### Command line arguments #####
	parser = argparse.ArgumentParser(description="Creates a cross-team localization report.")

	addDataGroup = parser.add_argument_group('Add Data', 'Options to create and add data to a database')
	queryDBGroup = parser.add_argument_group('Query DB', 'Options for querying an already existing database and outputting to HTML')

	#### Options for adding data from CSV to database ####
	addDataGroup.add_argument('--addOnly', '-a', action='store_true',
		help='Use this option to add CSV files to database without producing an HTML file')
	addDataGroup.add_argument('--csvFilePath', '-fp', type=str, default='None', nargs='+',
		help='Filepath to CSV containing score results: [e.g. ~/DryRunScores/HW_NC17_DRYRUN17_Manipulation_ImgOnly_p-baseline_1-mask_scores_perimage.csv]. If none given, will query already existing data in the database.')
	addDataGroup.add_argument('--expName', '-en', type=str, default='None', nargs='+',
		help='Experiment Name associated with respective csvFilePath [e.g. HW_p-baseline_1]')
	addDataGroup.add_argument('--addDir', '-ad', type=str, default=['None', 'Manipulation*perimage.csv'], nargs=2,
		help='Use this to add all CSVs in a specified directory with glob pattern [e.g. ~/DryRunScores/ *Manipulation*perimage.csv]. This will automatically generate experiment names in the form of <TEAM>_<SYS>_<VERSION> [e.g. HW_p-baseline_1]')
	addDataGroup.add_argument('--fixFilePath', '-ff', type=str, default=['None','None'], nargs=2,
		help='Use this option to correct filepaths of images in CSV file before data is inserted into database [e.g. /oldDirectory/ /newDirectory/]')
	addDataGroup.add_argument('--delimiter', '-d', type=str, default='|',
		help='Use to specify if your data files are not delimited by | (pipes). [e.g. ,]')

	#### Options for querying database and display of HTML page ####
	queryDBGroup.add_argument('--queryProbe', '-qp', type=str, default='all', nargs='+',
		help='ProbeIDs to be displayed in HTML file: [e.g. 003eaa9f0f222263550b95e0ab994f33]. If not used, defaults to creating a report with all probes.')
	queryDBGroup.add_argument('--queryExp', '-qe', type=str, default='all', nargs='+',
		help='Experiments to be displayed in HTML file: [e.g. HW_p-baseline_1]. Default is for all experiments. Experiment names must exist in database already and name must match exactly, or be added with --csvFilePath and --expName.')
	queryDBGroup.add_argument('--queryMCC', '-qm', type=str, default='all', nargs='+',
		help='Use to query MCC, numbers must be between. [-1, 1]: [e.g. "> 0.5"]. Multiple arguments will be ANDed together. [e.g. "> 0" "< 0.5" is equivalent to MCC > 0 AND MCC < 0.5] Default will query all MCCs of all values.')
	queryDBGroup.add_argument('--sortProbes', '-sp', type=str, default='None',
		help='Use to sort output results by probes based on descending (desc) or ascending (asc) order of max MCC for each probe [e.g. desc]. Default is none.')
	queryDBGroup.add_argument('--sortTeams', '-st', type=str, default='None',
		help='Use to sort output results by on descending (desc) or ascending (asc) order average MCC scores of each time [e.g desc]. Default is none.')
	queryDBGroup.add_argument('--outputFileName', '-o', type=str, default='output.html',
		help='Name of output file: [e.g. AllDryRunResults.html]. Defaults to output.html')
	queryDBGroup.add_argument('--highlightMax', '-hl', action='store_true',
		help='Hightlights the image/MCC for the mask with the highest MCC for each probe')

	#### Options for both adding to database and querying database ####
	parser.add_argument('--dbName', '-db', type=str, default='ScoresPerImage.db',
		help='Name of database to be created and/or queried: [e.g. DryRunScores.db]. Defaults to ScoresPerImage.db')
	parser.add_argument('--tableName', '-tn', type=str, default='CombinedCSVData',
		help='Name of table to be created and/or queried: [e.g. MaskScoresPerImage]. Defaults to CombinedCSVData')
	parser.add_argument('--verbose', '-v', action='store_true',
		help='Control print output.')

	#### Option for Jon ####
	parser.add_argument('--preQuery', '-pq', type=str, default='None')

	args = parser.parse_args()

	#### Error handling on incorrect input ####
	if (len(args.csvFilePath) != len(args.expName)):
		parser.error('--csvFilePath requires an argument with the --expName option for each argument provided in --csvFilePath.')
	if args.sortProbes not in ['None', 'asc', 'desc']:
		parser.error('--sortProbes argument must be asc or desc')
	if args.sortTeams not in ['None', 'asc', 'desc']:
		parser.error('--sortTeams argument must be asc or desc')
	if args.addOnly:
		if (args.csvFilePath == 'None' and args.addDir[0] == 'None'):
			parser.error('If using --addOnly, must provide a csv filepath (--csvFilePath ~/DryRunScores/HW_NC17_DRYRUN17_Manipulation_ImgOnly_p-baseline_1-mask_scores_perimage.csv) and associated experiment name (--expName HW_p-baseline_1) or a directory with a matching pattern (--addDir ~/DryRunScores/ *Mani*perimage.csv)')

	tableName = args.tableName

	# Inserts data from each CSV file and associates it with respective experiment name
	if (args.csvFilePath > 0 and args.csvFilePath != 'None'):
		for dataFilePath, expName in zip(args.csvFilePath, args.expName):
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
			if args.verbose:
				print(createTable(args.dbName, args.preQuery))
			else:
				createTable(args.dbName, args.preQuery)

		generalInfo, perExpInfo = queryDB(args.dbName, tableName, args.queryProbe, args.queryExp, args.queryMCC, args.verbose)
		
		# Output overallResults to an HTML file
		if args.verbose:
			print(htmlReport(generalInfo, perExpInfo, args.outputFileName, args.sortProbes, args.sortTeams, args.highlightMax))
		else:
			htmlReport(generalInfo, perExpInfo, args.outputFileName, args.sortProbes, args.sortTeams, args.highlightMax)

	sys.exit(0)


def createTable(db, query):
	# Should be used to create a table in the database that will be used for HTML output

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

	try:
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
						try:
							c.execute(sqlText, insertRow)
						except sqlite3.OperationalError:
							return 'Check delimiters in CSV file, %s. If not |, use --delimiter <delimiter type> (e.g. --delimiter ,).' % CSVFilePath
			
			except csv.Error, e:
				return 'File %s was unable to be loaded into the DB' % CSVFilePath
	except IOError:
		return 'File %s was unable to open. Check name and/or filepath.' % CSVFilePath

	conn.commit()
	conn.close()

	return '%s successfully added' % expName


def queryDB(database, tableName, probes, exps, MCC, verbose):
	# Query database based on user input: 
	#	If no input, queries all probes
	#	If only probes specified, queries all experiments for specified probes
	#	If only experiments specified, queries all probes for specified experiments
	#	If probes and experiments specified, queries specified probes for specified experiments
	#   If probes, experiments, and MCC specified, queries based on those parameters
	
	conn = sqlite3.connect(database)
	c = conn.cursor()

	# List of elements that I want to query that are the same among all experiements
	queryElements = ['ProbeFileID', 'AggMaskFileName', 'ProbeMaskFileName']

	# List of elements that are unique to each experiment
	# Using cast(MCC as real) instead of just MCC to handle the case where MCC gets stored in scientific notation (e-05)
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

	for probe in probesList:

		# Progress output
		if verbose:
			count += 1
			percentage = float(count) / total * 100
			msgText = "Querying is %.2f%% complete" % percentage
			print msgText, "               \r",

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

			# If no data is found for a given probe and/or experiment name	
			if results == None and isScored == None:
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

def htmlReport(general, perExp, outputFile, sortProbes, sortTeams, highlightMax):
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

			for exp in unpackExps:
				total = 0
				count = 0
				for probe in exp:
					if probe[1] != None:
						total += probe[1]
						count += 1
				if count == 0:
					exp.insert(0, 0)
				else:		
					exp.insert(0, float(total)/count)

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

			# Finds the max MCC for each probe
			maxVal = 0
			if highlightMax:
				maxVal = max([sublist[1] for sublist in exps])
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
					# Highlights max value for each probe
					if experiment[1] == maxVal and highlightMax:
						experimentResults = """<td bgcolor="FFBB33"><b>MCC: %s</b><br>
						<a class="thumb" href="#"><img src="%s" alt="Evaluation Results" height="228px" width="304px">
						<span><img src="%s" alt="" height="100%%" width="100%%"></span></a></td>
						""" % (str(experiment[1]), experiment[3], experiment[3])
					else:
						experimentResults = """<td>MCC: %s<br>
						<a class="thumb" href="#"><img src="%s" alt="Evaluation Results" height="228px" width="304px">
						<span><img src="%s" alt="" height="100%%" width="100%%"></span></a></td>
						""" % (str(experiment[1]), experiment[3], experiment[3])
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

					# HTML template for each row
					result = """<tr>
						<td>%s<br><a class="thumb" href="#"><img src="%s" alt="Compositie Mask with Color" height="228px" width="304px">
						<span><img src="%s" alt="" height="100%%" width="100%%"></span></a></td>
 						<td><br><a class="thumb" href="#"><img src="%s" alt="Binarized Reference Mask" height="228px" width="304px">
 						<span><img src="%s" alt="" height="100%%" width="100%%"></span></a></td>
 						%s
					</tr>""" % (result[0], result[1], result[1], '/'.join(binRefMaskFilePathList), '/'.join(binRefMaskFilePathList), expsJoined)

					resultsList.append(result)
					break

		# Puts the pieces of HTML together and formats HTML based on 'outer' template
		headersJoined = '<th>' + '</th>\n\t\t<th>'.join(headers) + '</th>'
		rowsJoined = ''.join(resultsList)

		# HTML outline for output
		htmlOut = """
		<!DOCTYPE html>
		<html>
		<head>
		<style>
		.thumb {
		float:left;
		position:relative;
		margin:3px;
		}
		.thumb table, th, td {
			border: 1px solid black;
		}
		.thumb td {
			height: 228px;
			width: 310px;
		}
		.thumb img { 
			vertical-align: bottom;
		}
		.thumb:hover {
			z-index: 1;
		}
		.thumb span { 
			position: absolute;
			visibility: hidden;
		}
		.thumb:hover span { 
			visibility: visible;
			top: 37px; left: 37px; 
			height: 800px;
			width: 800px;
			border: 3px solid purple;
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