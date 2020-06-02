pull_from_server(){
    rsync -rave "ssh -i ~/Documents/AWS/frankly.pem.txt" ec2-user@$1:~/SchemaPrediction/json_files/* ./json_files/
}

status_check(){
    ssh -i ~/Documents/AWS/frankly.pem.txt ec2-user@$servername "ps -ef | grep python"
    scp -i ~/Documents/AWS/frankly.pem.txt -r ec2-user@$1:~/SchemaPrediction/batch_runner.log ./batch_runner_$2.log
}


while :
do

    echo "Attempting to download"
    # Server A
    servername="ec2-3-20-206-209.us-east-2.compute.amazonaws.com"
    pull_from_server $servername
    status_check $servername 0 
     
    # Server B
    servername="ec2-3-18-107-245.us-east-2.compute.amazonaws.com"
    pull_from_server $servername
    status_check $servername 1

    # Server C
    servername="ec2-18-191-7-223.us-east-2.compute.amazonaws.com"
    pull_from_server $servername
    status_check $servername 2

    # Server D
    servername="ec2-18-222-239-223.us-east-2.compute.amazonaws.com"
    pull_from_server $servername
    status_check $servername 3
    
    printf "\nDone!\n"

    t=10
    while [ $t -gt 0 ]
    do
        # message = "Waiting %\d minutes before next attempt" t
        printf "Waiting %d minutes before next attempt\n" $t
        ((t--))
        sleep 60s
    done

done
