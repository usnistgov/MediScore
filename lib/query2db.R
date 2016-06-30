# *File: query2db.r
# *Date: 5/26/2016
# *Author: Daniel
# *Status: Complete
#
# *Description: this code queries a database through MySQL 
#
#* Inputs
#    * q: a string for the query to send to the database
#    * db: the name of the database
#
#* Outputs
#    * myQuery: the results of the query
#
# *Disclaimer: 
# This software was developed at the National Institute of Standards 
# and Technology (NIST) by employees of the Federal Government in the
# course of their official duties. Pursuant to Title 17 Section 105 
# of the United States Code, this software is not subject to copyright 
# protection and is in the public domain. NIST assumes no responsibility 
# whatsoever for use by other parties of its source code or open source 
# server, and makes no guarantees, expressed or implied, about its quality, 
# reliability, or any other characteristic."
#################################################################

suppressMessages(require(RMySQL))

queryToDb <- function(q,db) {
  tryCatch({
    mydb <- dbConnect(MySQL(), user='simo', password='123', dbname=db, host='borg07.ncsl.nist.gov')
    myQuery <- dbGetQuery(mydb,q)
  }, error = function(e) {
    printq(paste("Error in query:",e))
  },
  finally={
    if (dbIsValid(mydb)) {
      dbDisconnect(mydb)
      return(myQuery)
    }
  }
  )
}
