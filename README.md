# Group-tsg

Model: 

![Model Architecture](group-tsg.png)

The definition of blocks of differnet types and backcast, forecast residual mechanism is the same with paper [N-BEATS](https://arxiv.org/abs/1905.10437)

Dataset: Tourism

1. Build docker image
    ```shell script
    make init
    ```

1. Download datasets
    ```shell script
    make dataset
    ```
   This command will download dataset into `./storage/datasets` directory

1. (Optional) Test metrics. To make sure that all datasets are correct and the metrics 
calculation works as expected you can run test.  
    ```shell script
    make test
    ```

1. Build an experiment
    ```shell script
    make build config=experiments/tourism/tourism.gin
    ```
   This will generate directories with configurations and command for each model of ensemble 
   in `./storage/experiments/group-tsg`. Note that the `config` parameter takes the **relative** 
   path to actual configuration.

1. Run experiments.
   Substitute different values for `repeat` and `lookback` in the command lines below to
   run other configurations of a model.
    
    CPU
    ```shell script
    make run command=storage/experiments/group-tsg/repeat=0,lookback=2,loss=MAPE/command
    ```
    GPU 
    ```shell script
     make run command=storage/experiments/group-tsg/repeat=0,lookback=2,loss=MAPE/command gpu=<gpu-id>
     ```
    If you have multiple GPUs on the same machine then run this command in parallel for each gpu-id.
   
    The logs, losses, snapshots and final forecasts will be stored in 
    `storage/experiments/group-tsg/repeat=0,lookback=2,loss=MAPE` directory.

    You can of course automate running across all experiments with the following example (assuming BASH):
    ```shell script
    for instance in `/bin/ls -d storage/experiments/group-tsg/*`; do 
        echo $instance
        make run command=${instance}/command
    done
    ```
