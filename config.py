from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
  team_name: str = 'voyager'
  sample_rate: int = 16000
  folder_name: str = "preprocessed_data"
  results_dir: str = "results"
  train_file_list: str = 'data/fmcc_train.ctl' 
  train_file_dir: str = '<YOUR_TRAIN_DATA_DIR>'
  test_file_list: str = 'data/fmcc_test_ref.txt' 
  test_file_dir: str = '<YOUR_TRAIN_DATA_DIR>'
  eval_file_list: str = 'data/fmcc_eval.ctl'
  eval_file_dir: str = '<YOUR_TRAIN_DATA_DIR>'
