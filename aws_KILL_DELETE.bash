KILL_DELETE(){
    ssh -i ~/Documents/AWS/frankly.pem.txt ec2-user@$1 "killall python"
    ssh -i ~/Documents/AWS/frankly.pem.txt ec2-user@$1 "rm -rf ~/SchemaPrediction/json_files/*"
}



while :
do

    echo "Attempting to download"
    # Server A
    servername="ec2-18-218-149-157.us-east-2.compute.amazonaws.com"
    KILL_DELETE $servername
     
    # Server B
    servername="ec2-3-15-169-104.us-east-2.compute.amazonaws.com"
    KILL_DELETE $servername

    # Server C
    servername="ec2-18-216-193-161.us-east-2.compute.amazonaws.com"
    KILL_DELETE $servername

    # Server D
    servername="ec2-18-216-81-81.us-east-2.compute.amazonaws.com"
    KILL_DELETE $servername
    
    printf "\nDone!\n"
    break;

done
