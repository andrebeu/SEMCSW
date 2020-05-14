status_check(){
    ssh -i ~/Documents/AWS/frankly.pem.txt ec2-user@$servername "ps -ef | grep python"
    scp -i ~/Documents/AWS/frankly.pem.txt -r ec2-user@$1:~/SchemaPrediction/batch_runner.log ./batch_runner_$2.log
}


# Server A
servername="ec2-18-218-149-157.us-east-2.compute.amazonaws.com"
status_check $servername 0
    
# Server B
servername="ec2-3-15-169-104.us-east-2.compute.amazonaws.com"
status_check $servername 1

# Server C
servername="ec2-18-216-193-161.us-east-2.compute.amazonaws.com"
status_check $servername 2

# Server D
servername="ec2-18-216-81-81.us-east-2.compute.amazonaws.com"
status_check $servername 3

printf "\nDone!\n"
