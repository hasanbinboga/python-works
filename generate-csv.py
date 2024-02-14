import csv
import random
from datetime import datetime, timedelta

def generate_random_data(start_date, end_date, interval_minutes):
    current_date = start_date
    data = []
    while current_date <= end_date:
        data.append([current_date.strftime("%d.%m.%Y %H:%M:%S"), random.randint(500, 9999)])
        current_date += timedelta(minutes=interval_minutes)
    return data

def write_to_csv(filename, data):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time', 'Value'])
        writer.writerows(data)

start_date = datetime(2023, 1, 1)
end_date = datetime.now()
interval_minutes = 15

random_data = generate_random_data(start_date, end_date, interval_minutes)
write_to_csv("Linux.csv", random_data)
