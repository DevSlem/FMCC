from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
  team_name: str = 'voyager'
  sample_rate: int = 16000
  folder_name: str = "preprocessed_data"
  train_file_list: str = 'fmcc_train.ctl' 
  train_file_dir: str = 'raw16k/train/'
  test_file_list: str = 'fmcc_test_ref.txt' 
  test_file_dir: str = 'raw16k/test/'
  eval_file_list: str = 'fmcc_test.ctl'
  eval_file_dir: str = 'raw16k/test/'
