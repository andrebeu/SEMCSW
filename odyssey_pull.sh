
run(){
    # ssh -i ~/Documents/AWS/frankly.pem.txt ec2-user@$1 "mkdir SchemaPrediction"
    # ssh -i ~/Documents/AWS/frankly.pem.txt ec2-user@$1 "mkdir SchemaPrediction/json_files"
    # scp -i ~/Documents/AWS/frankly.pem.txt -r ~/Projects/SchemaPrediction/*py ec2-user@$1:~/SchemaPrediction/
    # scp -i ~/Documents/AWS/frankly.pem.txt -r ~/Projects/SchemaPrediction/requirements.txt ec2-user@$1:~/SchemaPrediction/
    # scp -i ~/Documents/AWS/frankly.pem.txt -r ~/Projects/SchemaPrediction/aws_run.bash ec2-user@$1:~/SchemaPrediction/
    echo "Transfering Files"
    # ssh -i ~/Documents/AWS/frankly.pem.txt $1@$2 "mkdir SchemaPrediction"
    # ssh -i ~/Documents/AWS/frankly.pem.txt $1@$2 "mkdir SchemaPrediction/json_files"
    # scp -r ~/Projects/SchemaPrediction/*py $1@$2:~/SchemaPrediction/
    # scp -r $1@$2:~/SchemaPrediction/json_files/* json_files_e1e-8/

# }
    # rsync -rav $1@$2:~/SchemaPrediction/json_files_v061020_online/ ./json_files_v061020_online/
    # rsync -rav $1@$2:~/SchemaPrediction/json_files_v061020/ ./json_files_v061020_highlr/
    # rsync -rav $1@$2:~/SchemaPrediction/json_files_v071420/ ./json_files_v071420_v4/
    rsync -rav $1@$2:~/SchemaPrediction/json_files_v080320/ ./json_files_v080720/
    rsync -rav $1@$2:~/SchemaPrediction/json_files_v071420_MLP/ ./json_files_v071420_MLP_mixed2/

}


username="nfranklin"
servername="login.rc.fas.harvard.edu"
run $username $servername