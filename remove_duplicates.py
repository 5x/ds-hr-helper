if __name__ == '__main__':
    input_file = 'it_work_ua.txt'
    output_file = 'it_jobs.txt'
    encoding = 'utf-8'

    with open(input_file, 'r', encoding=encoding) as file:
        data = file.readlines()

    with open(output_file, 'a', encoding=encoding) as file:
        file.writelines(list(set(data)))
