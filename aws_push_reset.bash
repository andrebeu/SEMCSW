reset(){
    scp -i ~/Documents/AWS/frankly.pem.txt -r ~/Projects/SchemaPrediction/*py ec2-user@$1:~/SchemaPrediction/
    scp -i ~/Documents/AWS/frankly.pem.txt -r ~/Projects/SchemaPrediction/aws_run_reset.bash ec2-user@$1:~/SchemaPrediction/
    echo "killing active python jobs..."
    ssh -i ~/Documents/AWS/frankly.pem.txt ec2-user@$1 "killall python"
    echo "Starting batch jobs"
    ssh -i ~/Documents/AWS/frankly.pem.txt ec2-user@$1 "./SchemaPrediction/aws_run_reset.bash $2"
}

# Server A
servername="ec2-18-218-149-157.us-east-2.compute.amazonaws.com"
batch_n=0
reset $servername $batch_n

# Server B
servername="ec2-3-15-169-104.us-east-2.compute.amazonaws.com"
batch_n=1
reset $servername $batch_n

# Server C
servername="ec2-18-216-193-161.us-east-2.compute.amazonaws.com"
batch_n=2
reset $servername $batch_n

# Server D
servername="ec2-18-216-81-81.us-east-2.compute.amazonaws.com"
batch_n=3
reset $servername $batch_n


./aws_pull.bash