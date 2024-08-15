import csv  
from pathlib import Path  


def get_time(file_path, time_use_list):
    with open(file_path, 'r', encoding='utf-8') as file:  
        reader = csv.DictReader(file)
        for row in reader:  
            time_use = row['Task Duration(us)']
            time_use_list.append(int(float(time_use)* 1000000))


def find_min_time():
    min_time = 0
    time_use_list = []
    directory = Path('./')
    filename_pattern = 'OpBasicInfo.csv'
    
    for file in directory.rglob(filename_pattern):  
        get_time(file, time_use_list)
    
    if len(time_use_list) > 0:
        min_time = min(time_use_list) 

    print(min_time)

if __name__ == '__main__':
    find_min_time()
    


