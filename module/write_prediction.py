file_name = open(f'202301ml_fmcc/fmcc_test_ref.txt', 'r')
test_file_name = []

for i in file_name.readlines():
    file, _ = i.strip('\n').split(' ')
    test_file_name.append(file)

def write_predictions(predictions, filename):
    with open(filename, 'w') as file:
        for index, pred in enumerate(predictions):
            file.write((f'{test_file_name[index]} ' + ('feml' if pred == 1 else 'male')) + '\n')