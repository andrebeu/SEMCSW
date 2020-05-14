conda create --name schema python=3.6
source activate schema
cd ~/SchemaPrediction
pip install -r requirements.txt
nohup python -u schema_prediction_batch_runner_05-12-20.py $1 &> batch_runner.log & 
sleep 10
