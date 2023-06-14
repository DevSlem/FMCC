# FMCC

Female-Male Classification Challenge

## Installation

If you use Anaconda, create an Anaconda environment first by entering the command below (optional):

```bash
conda create -n fmcc python=3.9 -y
conda activate fmcc
```

Install packages:

```bash
pip install ipykernel==6.23.1
pip install scikit-learn==1.2.2
pip install matplotlib==3.7.1
```

## Run

**Enter the following commands to run:**

```shell
python main.py # train
python main.py -e # evaluation
perl eval.pl voyager_test_results.txt fmcc_test_ref.txt # print accuracy
```

**Detail Options:**

```
Usage:
    python main.py [options]

Options:
    -p --plot                     Whether to plot [default: False].
    -i --inference                Whether to inference or test [default: False].
    -e --eval                     Whether to evaluate [default: False].

Config:
    train_file_list               train 데이터 리스트 정보가 담긴 파일
    test_file_list                test 데이터 리스트 정보가 담긴 파일
    eval_file_list                eval 데이터 리스트 정보가 담긴 파일

    train_file_dir                train 음성파일이 있는 디렉터리명
    test_file_dir                 test 음성파일이 있는 디렉터리명
    eval_file_dir                 eval 음성파일이 있는 디렉터리명
```

- During the evaluation, the gender classification results for male and female voices are generated from unlabeled data and saved in the file `voyager_test_results.txt`.
- During inference, using the given test data, you can obtain results such as accuracy and other metrics.
- The trained model is saved in the `saved_model` file during training.
- Set the files for evaluation by configuring the `eval_file_dir` and `eval_file_list` in the `config.py`.
