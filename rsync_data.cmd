gsname='sem1'
echo "RSYNC GRIDSEARCH DATA FROM:"
echo ${gsname}
echo
rsync -r -P -vam abeukers@scotty.princeton.edu:/jukebox/norman/abeukers/sem/SchemaPrediction_internal/gsdata/${gsname}/* /Users/abeukers/wd/csw/SchemaPrediction_internal/gsdata/${gsname}