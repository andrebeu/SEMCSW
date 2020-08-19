
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
    rsync -a requirements.txt *py *.sh --exclude="*.json" $1@$2:~/SchemaPrediction/
}

username="nfranklin"
servername="login.rc.fas.harvard.edu"
run $username $servername