gsname='absem'
echo "RSYNC GRIDSEARCH DATA FROM:"
echo ${gsname}
echo
rsync -r -vam --progress abeukers@scotty.princeton.edu:/jukebox/norman/abeukers/sem/SchemaPrediction_internal/gsdata/${gsname}/* /Users/abeukers/wd/csw/SchemaPrediction_internal/gsdata/${gsname}