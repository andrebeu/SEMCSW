reset(){
    scp -i ~/Documents/AWS/frankly.pem.txt -r ~/Projects/SchemaPrediction/*py ec2-user@$1:~/SchemaPrediction/
    scp -i ~/Documents/AWS/frankly.pem.txt -r ~/Projects/SchemaPrediction/aws_run_reset.bash ec2-user@$1:~/SchemaPrediction/
    echo "killing active python jobs..."
    ssh -i ~/Documents/AWS/frankly.pem.txt ec2-user@$1 "killall python"
    echo "Starting batch jobs"
    ssh -i ~/Documents/AWS/frankly.pem.txt ec2-user@$1 "./SchemaPrediction/aws_run_reset.bash $2"
}

# Server A
servername="ec2-3-20-206-209.us-east-2.compute.amazonaws.com"
batch_n=0
reset $servername $batch_n

# Server B
servername="ec2-3-18-107-245.us-east-2.compute.amazonaws.com"
batch_n=1
reset $servername $batch_n

# Server C
servername="ec2-18-191-7-223.us-east-2.compute.amazonaws.com"
batch_n=2
reset $servername $batch_n

# Server D
servername="ec2-18-222-239-223.us-east-2.compute.amazonaws.com"
batch_n=3
reset $servername $batch_n

# # Server E
# servername="ec2-18-217-250-118.us-east-2.compute.amazonaws.com"
# batch_n=4
# reset $servername $batch_n

# # Server F
# servername="ec2-18-216-81-227.us-east-2.compute.amazonaws.com"
# batch_n=5
# reset $servername $batch_n

# # Server G
# servername="ec2-18-222-187-9.us-east-2.compute.amazonaws.com"
# batch_n=6
# reset $servername $batch_n

# # Server H
# servername="ec2-18-217-48-5.us-east-2.compute.amazonaws.com"
# batch_n=7
# reset $servername $batch_n

./aws_pull.bash