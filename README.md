# FMCC

Female-Male Classification Challenge

1. 학습용 코드 `module/train.py`
2. 모델 `results/saved_model`
3. 테스트용 코드 `module/test.py`
4. 테스트 결과 파일 `results/voyager_test_results.txt`
5. 실행 설명서 `README.md`
6. 논문 형식 결과 보고서 `fmcc_report.hwp`

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
perl eval.pl results/voyager_test_results.txt data/fmcc_test_ref.txt # print accuracy
```

**Detail Options:**

```
Usage:
    python main.py [options]

Options:
    -p --plot                     Whether to plot [default: False].
    -i --inference                Whether to inference or test [default: False].
    -e --eval                     Whether to evaluate [default: False].
```

- During the evaluation, the gender classification results for male and female voices are generated from unlabeled data and saved in the file `voyager_test_results.txt`.
- During inference, using the given test data, you can obtain results such as accuracy and other metrics.
- The trained model is saved in the `saved_model` file during training.

```
Config:
    train_file_list               file containing information about the train data list.
    test_file_list                file containing information about the test data list.
    eval_file_list                file containing information about the evaluation data list.

    train_file_dir                directory name where the train audio files are located.
    test_file_dir                 directory name where the test audio files are located.
    eval_file_dir                 directory name where the evaluation audio files are located.
```

- If you run evaluation mode (by entering `python main.py -e`), set the files for evaluation by configuring the `eval_file_dir` and `eval_file_list` in the `config.py`.
