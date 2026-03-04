Project description

nnunet_trainer is a long running worker. 

At the start, it register itself to nnunet_dashboard.
Periodically check if there is any assigned training.
If yes, then download the dataset, which includes the raw dataset as well as the plan. Do preprocessing train for the requested configuration. 

Please refer to the project @nnUNet (pyproject.yml) for the nnunet commands to do preprocess and train for a given configuration, as well as the project nnunet_server, which knows how to submit jobs for preprocessing (no planning is needed since the dataset already contains the plan), and how to train. Either use the rq with redis as a queue system, or simply spine off child proceses to run the commands, and monitor the progress by looking at the log files and make the status reports to nnunet_dashboard via api.

for the project, create a conda environment namee 'nnunet_trainer', and install nnunetv2 and other necessary dependent packages.

Once the job is complete, check the dashboard periodically to find the next assigned job to run.
