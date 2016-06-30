#!/bin/bash

echo 'Creating tables...'
Rscript ../../../../tools/Medifor_SetupScoringDB.r mediforDaniel \
&& echo 'Enrolling team...' && Rscript ../../../../tools/Medifor_EnrollTeam.r -t foo -d mediforDaniel \
&& echo 'Validating submission and populating database...' \
&& Rscript ../../../../tools/SSD_Validate.r -s ../foo_NC2016_Manipulation_ImgOnly_p-baseline_1/foo_NC2016_Manipulation_ImgOnly_p-baseline_1.csv -x indexes/NC2016-manipulation-index.psv -d mediforDaniel
