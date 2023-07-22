from finetune.finetune_tasks import Data_Compensation, Data_Prediction, Data_Simulation

data_dict = {
    'compensation': Data_Compensation,
    'prediction': Data_Prediction,
    'simulation': Data_Simulation,
}

def gen_dataset(config, mark="compensation"):
    Data = data_dict[mark]
    data_set = Data(index_path=config.index_path,
                    data_path=config.data_path,
                    interval=config.interval,
                    max_seq_len=config.max_seq_len)
    return data_set
