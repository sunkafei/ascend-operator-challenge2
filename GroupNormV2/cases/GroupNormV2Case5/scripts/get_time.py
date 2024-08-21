import csv  
from pathlib import Path  
from os import popen

def get_time(file_path, time_use_list):
    with open(file_path, 'r', encoding='utf-8') as file:  
        reader = csv.DictReader(file)
        for row in reader:  
            time_use = row['Task Duration(us)']
            time_use_list.append(int(float(time_use)*1000))


def find_min_time():
    min_time = 0
    time_use_list = []
    directory = Path('./')
    filename = popen('find ./ -name op_summary*.csv').read().strip()

    get_time(filename, time_use_list)
    
    if len(time_use_list) > 0:
        min_time = min(time_use_list) 

    print(min_time)

if __name__ == '__main__':
    find_min_time()
    


