import os, time

def make_slurm_shell(filename="_slurm.sh"):

    # these are the file lines we will have
    lines = [
        "#!/bin/bash",
        "#SBATCH -p shared # partition (queue)",
        "#SBATCH -N 1 # number of nodes",
        "#SBATCH -n 1 # number of cores",
        "#SBATCH --mem 3000 # memory pool for all cores",
        "#SBATCH -t 0-4:00 # time (D-HH:MM)",
        "#SBATCH -o slurm.%N.%j.out # STDOUT",
        "#SBATCH -e slurm.%N.%j.err # STDERR",
        "",
        "module load Anaconda3/2019.10",
        "conda create -n schema",
        "source activate schema",
        "cd ~/SchemaPrediction",
        "pip install -r requirements.txt",
        "python -u schema_prediction_batch_runner_06-02-20.py &> ./logs/batch_submitter.log",
        "sleep 10",
    ]

    with open(filename, 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')
        f.close()
    
if __name__ == "__main__":

    filename = '_slurm.sh'
    make_slurm_shell(filename=filename)
    os.system('sbatch _slurm.sh')

