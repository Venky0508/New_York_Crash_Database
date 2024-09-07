"""
CSCI-720 Project:

Data Mining Project to extract useful information from the New York City Vehicle Collisions dataset.

author: Abhinav Ajit Menon (am6176)
        Srivenkatesh Shivadas Nair (sn6711)
"""
import argparse
from datetime import datetime
from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import folium
from folium.plugins import DualMap
from sklearn.cluster import DBSCAN


def read_csv_file(csv_file_path):
    """
    Read a single file from the path provided
    :param csv_file_path: file path name
    :return: the dataframe
    """
    try:
        df = pd.read_csv(csv_file_path)
        return df
    except Exception as e:
        print(f"Error reading file {csv_file_path}: {e}")
        return None


def filter_data(data):
    """
    Filter & clean the data and extract data points for Bronx for the year 2019-2020
    :param data: the dataframe
    :return: data_borough
    """

    data['CRASH DATE'] = pd.to_datetime(data['CRASH DATE'], errors='coerce')  # Convert to datetime

    # Filter data based on date range
    start_date = pd.to_datetime('01-01-2019')
    end_date = pd.to_datetime('12-30-2020')
    filtered_data = data[(data['CRASH DATE'] >= start_date) & (data['CRASH DATE'] <= end_date)]

    data_borough = filtered_data[filtered_data['BOROUGH'] == 'BRONX']
    data_borough = data_borough.drop(['BOROUGH'], axis=1)

    data_borough['CONTRIBUTING FACTOR VEHICLES'] = data_borough.apply(
        lambda row: [row['CONTRIBUTING FACTOR VEHICLE 1'],
                     row['CONTRIBUTING FACTOR VEHICLE 2'],
                     row['CONTRIBUTING FACTOR VEHICLE 3'],
                     row['CONTRIBUTING FACTOR VEHICLE 4'],
                     row['CONTRIBUTING FACTOR VEHICLE 5']], axis=1)

    data_borough = data_borough.drop(
        ['CONTRIBUTING FACTOR VEHICLE 1', 'CONTRIBUTING FACTOR VEHICLE 2', 'CONTRIBUTING FACTOR VEHICLE 3',
         'CONTRIBUTING FACTOR VEHICLE 4', 'CONTRIBUTING FACTOR VEHICLE 5'], axis=1)

    data_borough['VEHICLE TYPE CODES'] = data_borough.apply(
        lambda row: [row['VEHICLE TYPE CODE 1'],
                     row['VEHICLE TYPE CODE 2'],
                     row['VEHICLE TYPE CODE 3'],
                     row['VEHICLE TYPE CODE 4'],
                     row['VEHICLE TYPE CODE 5']], axis=1)
    data_borough = data_borough.drop(
        ['VEHICLE TYPE CODE 1', 'VEHICLE TYPE CODE 2', 'VEHICLE TYPE CODE 3', 'VEHICLE TYPE CODE 4',
         'VEHICLE TYPE CODE 5'], axis=1)

    return data_borough


def summer_visualization(data):
    """
    Count the total number of Injuries and Fatalities in the year 2019 and 2020.
    :param data: the dataframe
    :return: injuries_2019, injuries_2020, fatalities_2019, fatalities_2020
    """
    data['CRASH DATE'] = pd.to_datetime(data['CRASH DATE'])
    summer_data_2019 = data[
        (data['CRASH DATE'].dt.month >= 6) & (data['CRASH DATE'].dt.month <= 8) & (data['CRASH DATE'].dt.year == 2019)]
    summer_data_2020 = data[
        (data['CRASH DATE'].dt.month >= 6) & (data['CRASH DATE'].dt.month <= 8) & (data['CRASH DATE'].dt.year == 2020)]

    # Total injuries and fatalities for each summer
    injuries_2019 = summer_data_2019['NUMBER OF PERSONS INJURED'].sum()
    injuries_2020 = summer_data_2020['NUMBER OF PERSONS INJURED'].sum()
    fatalities_2019 = summer_data_2019['NUMBER OF PERSONS KILLED'].sum()
    fatalities_2020 = summer_data_2020['NUMBER OF PERSONS KILLED'].sum()

    print(f"\nQuestion1: Comparison between summers of 2019-2020:\n")
    print(f"Injuries in 2019: {injuries_2019}, Injuries in 2020: {injuries_2020}")
    print(f"Fatalities in 2019: {fatalities_2019}, Fatalities in 2020: {fatalities_2020}")

    return injuries_2019, injuries_2020, fatalities_2019, fatalities_2020


def summer_plot(injuries_2019, fatalities_2019, injuries_2020, fatalities_2020):
    """
    A bar graph highlighting the number of injuries and fatalities between the years 2019 and 2020
    :param injuries_2019: Number of people injured in 2019
    :param fatalities_2019: Number of people killed in 2019
    :param injuries_2020: Number of people injured in 2020
    :param fatalities_2020: Number of people killed in 2020
    :return:
    """
    labels = ['Injuries', 'Fatalities']
    values_2019 = [injuries_2019, fatalities_2019]
    values_2020 = [injuries_2020, fatalities_2020]
    x = range(len(labels))

    plt.bar(x, values_2019, width=0.4, label='2019')
    plt.bar([i + 0.4 for i in x], values_2020, width=0.4, label='2020')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.title('Comparison of Injuries and Fatalities between Summers (2019 vs. 2020)')
    plt.xticks([i + 0.2 for i in x], labels)
    plt.legend()
    plt.show()


def clean_list(lst):
    """
    Function to clean a single list and filter the data for the DBscan algorithm
    :param lst:
    :return:
    """
    return [value for value in lst if pd.notna(value) and value != 'Unspecified']

def combine_months(data, year):
    """
    Combine the data from the months june, july and august to get the complete summer dataset.
    :param data: the dataframe
    :param year: the summer year
    :return:
    """
    june_data = data[
        (data['CRASH DATE'].dt.year == year) &
        (data['CRASH DATE'].dt.month == 6)
        ]
    july_data = data[
        (data['CRASH DATE'].dt.year == year) &
        (data['CRASH DATE'].dt.month == 7)
        ]
    august_data = data[
        (data['CRASH DATE'].dt.year == year) &
        (data['CRASH DATE'].dt.month == 8)
        ]

    return pd.concat([june_data, july_data, august_data])


def db_scan(data, year):
    """
    Cluster the data points in the dataset using the DBscan algorithm
    :param data: The dataframe
    :param year: the summer year
    :return:
    """

    data.dropna(subset=['LATITUDE', 'LONGITUDE'], inplace=True)
    X = data[['LATITUDE', 'LONGITUDE']].values

    if year == '2019':
        epsilon = 0.008
        min_samples = 2
    elif year == '2020':
        epsilon = 0.01
        min_samples = 4

    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples, metric='euclidean')

    clusters = dbscan.fit_predict(X)

    plt.scatter(X[:, 1], X[:, 0], c=clusters, cmap='viridis', marker='o', s=30)
    plt.title(f'DBSCAN Clustering of Summer {year} Data')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()


def filter_and_drop(data, start_date, end_date):
    """
    Filter the data on the basis of Latitude and Longitude
    :param data: The dataframe
    :param start_date: start date
    :param end_date: end date
    :return: filtered_data
    """
    filtered_data = data.loc[
        (data['CRASH DATE'] >= start_date) &
        (data['CRASH DATE'] <= end_date), ['LATITUDE', 'LONGITUDE']
    ].dropna()
    return filtered_data


def get_location_counts(data):
    """
    Count the number of accidents at each Location(Latitude, Longitude)
    :param data: the dataframe
    :return: location_counts
    """
    location_counts = data.groupby(['LATITUDE', 'LONGITUDE']).size().reset_index(name='ACCIDENT_COUNT')
    return location_counts


def q2_and_q3_folium(data):
    """
    Use Folium's DualMap plugin to plot the comparison for summer 2019-2020
    :param data: the dataframe
    :return:
    """
    m_june = folium.plugins.DualMap(
        location=[40.840508, -73.855150],
        zoom_start=12
    )

    m_july = folium.plugins.DualMap(
        location=[40.840508, -73.855150],
        zoom_start=12
    )

    june_2019_data = filter_and_drop(data, '2019-06-01', '2019-06-30')
    june_2020_data = filter_and_drop(data, '2020-06-01', '2020-06-30')

    july_2019_data = filter_and_drop(data, '2019-07-01', '2019-07-30')
    july_2020_data = filter_and_drop(data, '2020-07-01', '2020-07-30')

    june_location_counts_2019 = get_location_counts(june_2019_data).rename(
        columns={'ACCIDENT_COUNT': 'JUNE_ACCIDENT_COUNT_2019'})
    june_location_counts_2020 = get_location_counts(june_2020_data).rename(
        columns={'ACCIDENT_COUNT': 'JUNE_ACCIDENT_COUNT_2020'})

    july_location_counts_2019 = get_location_counts(july_2019_data).rename(
        columns={'ACCIDENT_COUNT': 'JULY_ACCIDENT_COUNT_2019'})
    july_location_counts_2020 = get_location_counts(july_2020_data).rename(
        columns={'ACCIDENT_COUNT': 'JULY_ACCIDENT_COUNT_2020'})

    for index, row in june_location_counts_2019.iterrows():
        radius = row['JUNE_ACCIDENT_COUNT_2019'] * 1
        folium.CircleMarker(
            location=[row['LATITUDE'], row['LONGITUDE']],
            radius=radius,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.6,
            popup=f"2019 Accident Count: {row['JUNE_ACCIDENT_COUNT_2019']}"
        ).add_to(m_june.m1)

    for index, row in june_location_counts_2020.iterrows():
        radius = row['JUNE_ACCIDENT_COUNT_2020'] * 1
        folium.CircleMarker(
            location=[row['LATITUDE'], row['LONGITUDE']],
            radius=radius,
            color='green',
            fill=True,
            fill_color='green',
            fill_opacity=0.6,
            popup=f"2020 Accident Count: {row['JUNE_ACCIDENT_COUNT_2020']}"
        ).add_to(m_june.m2)

    for index, row in july_location_counts_2019.iterrows():
        radius = row['JULY_ACCIDENT_COUNT_2019'] * 1
        folium.CircleMarker(
            location=[row['LATITUDE'], row['LONGITUDE']],
            radius=radius,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.6,
            popup=f"2019 Accident Count: {row['JULY_ACCIDENT_COUNT_2019']}"
        ).add_to(m_july.m1)

    for index, row in july_location_counts_2020.iterrows():
        radius = row['JULY_ACCIDENT_COUNT_2020'] * 1
        folium.CircleMarker(
            location=[row['LATITUDE'], row['LONGITUDE']],
            radius=radius,
            color='green',
            fill=True,
            fill_color='green',
            fill_opacity=0.6,
            popup=f"2020 Accident Count: {row['JULY_ACCIDENT_COUNT_2020']}"
        ).add_to(m_july.m2)

    m_june.save('june_2019-2020_accident_comparison.html')
    m_july.save('july_2019-2020_accident_comparison.html')


def plot1_q4(x, y):
    """
    Line graph indicating 100 consecutive days with most accidents.
    :param x: start dates
    :param y: Total accidents in that period
    :return:
    """
    plt.figure(figsize=(10, 7))
    plt.plot(x, y, color='purple', linewidth=2)
    plt.xlabel('Start Dates')
    plt.ylabel('Total Accidents in that Period')
    plt.title('100 Consecutive Days from Jan 2019 - Oct 2020 (Most Crashes)')
    plt.show()


def plot2_q4(period, data):
    """
    Count Plot indicating 100 consecutive days with most accidents.
    :param period: Time period for the crashes
    :param data: Dates
    :return:
    """
    plt.figure(figsize=(10, 7))
    sns.countplot(x=data, palette='viridis')
    plt.xticks(rotation=90, ha='center', fontsize=5)
    plt.xlabel('Dates')
    plt.ylabel('Crash Count')
    plt.title(f'Top 100 Consecutive Days with Maximum Crashes ({period[0]} - {period[-1]})')
    plt.show()


def q4(data):
    """
    Function to find 100 consecutive dates with most number of accidents.
    :param data: The dataframe
    :return:
    """
    date_counts_dict = data['CRASH DATE'].value_counts().reset_index()
    date_counts_dict.columns = ['Date', 'Count']
    date_counts_dict['Date'] = date_counts_dict['Date'].dt.strftime('%Y-%m-%d')
    date_counts_dict = dict(zip(date_counts_dict['Date'], date_counts_dict['Count']))

    start_date = datetime.strptime(min(date_counts_dict.keys()), '%Y-%m-%d')
    end_date = datetime.strptime(max(date_counts_dict.keys()), '%Y-%m-%d')
    date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]

    # Update the dictionary with missing dates and assign a default count of 0
    for date in date_range:
        date_str = date.strftime('%Y-%m-%d')
        if date_str not in date_counts_dict:
            date_counts_dict[date_str] = 0

    date_counts_sorted = sorted(date_counts_dict.items())
    x_vals = []
    y_vals = []

    current_sum = sum([date_counts_dict[date] for date, _ in date_counts_sorted[:100]])

    max_sum = current_sum
    max_period = [date for date, _ in date_counts_sorted[:100]]
    x_vals.append(max_period[0])
    y_vals.append(current_sum)

    # Loop through every 100 consecutive days
    for i in range(1, len(date_counts_sorted) - 100):
        current_sum = current_sum - date_counts_dict[date_counts_sorted[i - 1][0]] + date_counts_dict[
            date_counts_sorted[i + 99][0]]
        y_vals.append(current_sum)
        new_period = [date for date, _ in date_counts_sorted[i:i + 100]]
        x_vals.append(new_period[0])
        if current_sum > max_sum:
            max_sum = current_sum
            max_period = new_period

    datapoints = []
    for i in max_period:
        value = date_counts_dict[i]
        temp = [i] * value
        datapoints += temp

    print(f"Time period for most accidents in 100 consecutive days: {max_period[0]} to {max_period[-1]}")
    print("Total accidents during this period:", max_sum)
    plot1_q4(x_vals, y_vals)
    plot2_q4(max_period, datapoints)
    return date_counts_dict

def plot_q5(x, y):
    """
    Bar Graph showing the number of accidents on each day of the week.
    :param x: the day of the week
    :param y: Number of accidents
    :return:
    """
    plt.figure(figsize= (10, 7))
    plt.bar(x, y,color='blue', edgecolor='black', linewidth=2)
    plt.xlabel('Days')
    plt.ylabel('Frequency of each day')
    plt.title('Day of the Week (Most Crashes)')
    plt.show()


def q5(date_counts_dict):
    """
    Function to calculate the day of the week with most accidents
    :param date_counts_dict: Dictionary with dates as keys and number of accidents on that day as value.
    :return:
    """

    date_counts_dict = {datetime.strptime(date, '%Y-%m-%d'): count for date, count in date_counts_dict.items()}
    accidents_by_day = {i: 0 for i in range(7)}

    # Sum the accidents for each day of the week
    for date, count in date_counts_dict.items():
        day_of_week = date.weekday()
        accidents_by_day[day_of_week] += count

    # Find the day of the week with the most accidents
    max_accidents_day = max(accidents_by_day, key=accidents_by_day.get)
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    vals = list(accidents_by_day.values())
    plot_q5(days, vals)
    most_accidents_day_name = days[max_accidents_day]
    print(f"The day of the week with the most accidents is {most_accidents_day_name}, with a total of {accidents_by_day[max_accidents_day]} accidents.")


def plot_q6(data):
    """
    Parzen Density plot find the hour of the day with most accidents
    :param data: List of crash times
    :return:
    """
    plt.figure(figsize= (10, 7))
    sns.kdeplot(data, color='red', label='Parzen Density Estimation', fill=True)
    plt.title('Parzen Density Estimation')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Frequency of Crashes on each hour')
    plt.legend()
    plt.show()


def q6(crash_times):
    """
    Function to find the hour of the day with most accidents
    :param crash_times: List of crash times
    :return:
    """
    hours = [time.split(':')[0] for time in crash_times]
    hour_counts = Counter(hours)
    sorted_hour_counts = dict(sorted(hour_counts.items(), key=lambda x: int(x[0])))
    times = list(sorted_hour_counts.keys())
    times = [(int(i) + 1) for i in times]
    time_count = list(sorted_hour_counts.values())
    data = []
    for q in range(len(times)):
        temp = [times[q]] * time_count[q]
        data += temp

    plot_q6(data)

    max_hour_count = max(hour_counts.values())
    most_accident_hours = [hour for hour, count in hour_counts.items() if count == max_hour_count]
    print(f"The hour(s) with the most accidents is/are: {', '.join(most_accident_hours)} o'clock.")


def plot_q7(x, y):
    """
    Bar Graph for 12 days with most accidents.
    :param x: Dates
    :param y: Counts
    :return:
    """
    plt.figure(figsize=(10, 7))
    plt.barh(x, y, color=(0.2, 0.4, 0.6), edgecolor='black', linewidth=2)
    plt.xlabel('Count of Crashes')
    plt.ylabel('Dates')
    plt.title('Top 12 Days with most crashes in 2020')
    plt.show()


def q7(date_counts_dict):
    """
    Functicion to calculate 12 days with most accidents
    :param date_counts_dict: Dictionary containing number of accidents on each day
    :return:
    """
    year_2020_dates = {date: count for date, count in date_counts_dict.items() if
                       datetime.strptime(date, '%Y-%m-%d').year == 2020}
    top_12_days_2020 = sorted(year_2020_dates.items(), key=lambda x: x[1], reverse=True)[:12]
    x = []
    y = []

    print("The 12 days in 2020 with the most accidents are:")
    for date, count in top_12_days_2020:
        print(f"{date}: {count} accidents")
        x.append(date)
        y.append(count)

    plot_q7(x, y)


def main():
    """
    The main function that reads a csv file and performs Itemset Mining, DBscan clustering algorithm, and Visualization
    on Accident reports for the years 2019-2020 in New York City.
    return:
    """

    parser = argparse.ArgumentParser(description="Read and analyze CSV data")
    parser.add_argument("filename", type=str, nargs="?", default="Motor_Vehicle_Collisions_-_Crashes.csv",
                        help="CSV file to be read")
    args = parser.parse_args()

    # Call the function to read the CSV file
    data = read_csv_file(args.filename)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    num_rows = data.shape[0]
    num_columns = data.shape[1]
    print("\nDataset Information:\n")
    print(f"The number of rows in the original DataFrame: {num_rows}")
    print(f"The number of columns in the original DataFrame: {num_columns}")

    # Filter the data according to the requirements for the analysis
    print("\nBronx Filtered Dataset Information:\n")
    data_borough = filter_data(data)
    num_columns = data_borough.shape[1]
    num_rows = data_borough.shape[0]
    print(f"The number of columns in the DataFrame is: {num_columns}")
    print(f"The number of rows in the DataFrame is: {num_rows}")

    # Question1: Comparison between summers of 2019-2020
    injuries_2019, injuries_2020, fatalities_2019, fatalities_2020 = summer_visualization(data_borough)
    summer_plot(injuries_2019, fatalities_2019, injuries_2020, fatalities_2020)

    # DBscan
    summer_2019 = combine_months(data_borough, 2019)
    summer_2020 = combine_months(data_borough, 2020)

    summer_2019['CONTRIBUTING FACTOR VEHICLES'] = summer_2019['CONTRIBUTING FACTOR VEHICLES'].apply(clean_list)
    summer_2019['VEHICLE TYPE CODES'] = summer_2019['VEHICLE TYPE CODES'].apply(clean_list)
    summer_2019 = summer_2019[~(summer_2019['CONTRIBUTING FACTOR VEHICLES'].apply(len) == 0) & ~(
                summer_2019['VEHICLE TYPE CODES'].apply(len) == 0)]

    summer_2020['CONTRIBUTING FACTOR VEHICLES'] = summer_2020['CONTRIBUTING FACTOR VEHICLES'].apply(clean_list)
    summer_2020['VEHICLE TYPE CODES'] = summer_2020['VEHICLE TYPE CODES'].apply(clean_list)
    summer_2020 = summer_2020[~(summer_2020['CONTRIBUTING FACTOR VEHICLES'].apply(len) == 0) & ~(
            summer_2020['VEHICLE TYPE CODES'].apply(len) == 0)]

    db_scan(summer_2019, '2019')
    db_scan(summer_2020, '2020')
    print("The DBscan clustering plots show the comparison between 2019-2020")
    print("The Itemset Mining Algorithm show the comparison between 2019-2020 (Note: Open and run file Menon_Nair_itemset_mining.py)")

    # Question 2 and 3: Difference in crashes between June 2019 and June 2020?
    print("\nQuestion 2 and 3: Comparison between June 2019 and July 2019:\n")
    q2_and_q3_folium(data_borough)
    print("The Folium plot shows the comparison between the number of accidents between June of 2019 and 2020. (Note: Open the 'june_2019-2020_accident_comparison.html' file in the browser)")
    print("The Folium plot shows the comparison between the number of accidents between July of 2019 and 2020. (Note: Open the 'july_2019-2020_accident_comparison.html' file in the browser)")

    # Question 4: Most accidents in 100 consecutive days from January 2019 to October 2020
    print("\nQuestion 4: Most accidents in 100 consecutive days from January 2019 to October 2020\n")
    data_borough_sorted = data_borough.sort_values(by='CRASH DATE')
    date_counts_dict = q4(data_borough_sorted)

    # Question 5: Day of the week with most accidents
    print("\nQuestion 5: Day of the week with most accidents:\n")
    print("The bar graph plot shows the number of accidents every single day of the week.")
    q5(date_counts_dict)

    # Question 6: Hour of the day with most accidents
    print("\nQuestion 6: Hour of the day with most accidents:\n")
    crash_times = data_borough['CRASH TIME'].tolist()
    q6(crash_times)

    # Question 7: 12 days with most accidents in the year 2020
    print("\nQuestion 7: 12 days with most accidents in the year 2020:\n")
    q7(date_counts_dict)


if __name__ == "__main__":
    main()
