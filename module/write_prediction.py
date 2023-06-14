from config import Config

def write_predictions(predictions, filename):
    file_name = Config.eval_file_list
    test_file_name = []

    with open(file_name, 'r') as file:
        for line in file:
            file = line.strip('\n')
            test_file_name.append(file)
    with open(filename, 'w', newline='\r\n') as file:
        for index, pred in enumerate(predictions):
            file.write((f'{test_file_name[index]} ' + ('feml' if pred == 1 else 'male')) + '\n')