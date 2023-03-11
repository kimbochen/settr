# Summarizing Emotion Triggers with TRansformers


## Useful Commands

### Connect to a Slurm Cluster Node

```bash
# Open a bash shell
salloc -p <partition> --gres=gpu:1 --mem=12g -t=3:00:00

# Run an interactive job
srun -p <partition> --gres=gpu:1 --mem=12g -t=3:00:00

# Run another job within the same node
srun --jobid=<job_id>

# Check job status
squeue -u $USER

# Show node status
sinfo

# Show node info
scontrol show nodes <node_hostname>

# Show partition info
scontrol show partition <partition_name>
```

### Connect to Jupyter Notebooks on Slurm Cluster Nodes

```bash
# Find the IP <ip> of the node
cat /etc/hosts

# Listen to all ports
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0

# On the local computer
ssh cs260 -NL 8888:<ip>:8888
```
