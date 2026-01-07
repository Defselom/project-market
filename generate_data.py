import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
from faker import Faker
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Initialisation
fake = Faker()
np.random.seed(42)
random.seed(42)

# Parameters
START_DATE = "2018-01-01"
END_DATE = "2024-12-31"
PRODUCTS = [f"P{str(i).zfill(4)}" for i in range(1, 101)]
CATEGORIES = ['Groceries', 'Toys', 'Electronics', 'Clothing', 'Furniture', 'Home Appliances', 'Sports', 'Beauty']
REGIONS = ['North', 'South', 'East', 'West', 'Central', 'International']
WEATHER_CONDITIONS = ['Sunny', 'Cloudy', 'Rainy', 'Snowy', 'Windy', 'Foggy']
SEASONS = ['Spring', 'Summer', 'Autumn', 'Winter']

ENTRIES_PER_DAY_MIN = 300
ENTRIES_PER_DAY_MAX = 800
HIGH_VOLUME_DAYS = ['Friday', 'Saturday', 'Sunday']
SPECIAL_EVENTS_MULTIPLIER = 3

def get_season(date):
    month = date.month
    if 3 <= month <= 5:
        return 'Spring'
    elif 6 <= month <= 8:
        return 'Summer'
    elif 9 <= month <= 11:
        return 'Autumn'
    else:
        return 'Winter'

def get_special_days(year):
    special_days = []
    special_days.append(datetime(year, 1, 1))
    special_days.append(datetime(year, 12, 25))
    special_days.append(datetime(year, 12, 26))
    special_days.append(datetime(year, 12, 31))
    special_days.append(datetime(year, 7, 1))
    special_days.append(datetime(year, 7, 4))

    november_first = datetime(year, 11, 1)
    first_friday = november_first + timedelta(days=(4 - november_first.weekday()) % 7)
    black_friday = first_friday + timedelta(weeks=3)
    special_days.append(black_friday)
    special_days.append(black_friday + timedelta(days=1))
    special_days.append(black_friday + timedelta(days=2))

    cyber_monday = black_friday + timedelta(days=3)
    special_days.append(cyber_monday)

    prime_day = datetime(year, 7, 15)
    special_days.append(prime_day)

    singles_day = datetime(year, 11, 11)
    special_days.append(singles_day)

    back_to_school = datetime(year, 8, 20)
    special_days.append(back_to_school)

    special_days.append(datetime(year, 4, 1))
    special_days.append(datetime(year, 9, 1))

    return set(special_days)

def get_day_type(date, special_days):
    if date in special_days:
        return 'special'
    elif date.weekday() >= 5:
        return 'weekend'
    else:
        return 'weekday'

def get_category_stats(category, year):
    base_stats = {
        'Groceries': {'price_mean': 45, 'price_std': 20, 'units_mean': 180, 'units_std': 120},
        'Toys': {'price_mean': 55, 'price_std': 25, 'units_mean': 150, 'units_std': 110},
        'Electronics': {'price_mean': 350, 'price_std': 200, 'units_mean': 80, 'units_std': 60},
        'Clothing': {'price_mean': 65, 'price_std': 35, 'units_mean': 200, 'units_std': 150},
        'Furniture': {'price_mean': 280, 'price_std': 150, 'units_mean': 60, 'units_std': 45},
        'Home Appliances': {'price_mean': 220, 'price_std': 120, 'units_mean': 70, 'units_std': 50},
        'Sports': {'price_mean': 120, 'price_std': 80, 'units_mean': 90, 'units_std': 70},
        'Beauty': {'price_mean': 40, 'price_std': 25, 'units_mean': 170, 'units_std': 130}
    }
    stats = base_stats[category].copy()
    inflation_rate = 1 + ((year - 2018) * 0.025)
    stats['price_mean'] *= inflation_rate
    stats['price_std'] *= inflation_rate
    growth_rate = 1 + ((year - 2018) * 0.05)
    stats['units_mean'] *= growth_rate
    stats['units_std'] *= growth_rate
    return stats

def generate_day_data(date, special_days):
    day_type = get_day_type(date, special_days)
    if day_type == 'special':
        daily_entries = np.random.randint(ENTRIES_PER_DAY_MIN * 2, ENTRIES_PER_DAY_MAX * SPECIAL_EVENTS_MULTIPLIER)
    elif day_type == 'weekend':
        daily_entries = np.random.randint(ENTRIES_PER_DAY_MIN * 2, ENTRIES_PER_DAY_MAX * 2)
    else:
        daily_entries = np.random.randint(ENTRIES_PER_DAY_MIN, ENTRIES_PER_DAY_MAX)

    data = []
    season = get_season(date)
    year = date.year

    product_category_map = {}
    for product in PRODUCTS[:daily_entries]:
        if product not in product_category_map:
            product_category_map[product] = random.choice(CATEGORIES)

    for product in PRODUCTS[:daily_entries]:
        category = product_category_map[product]
        region = random.choice(REGIONS)
        stats = get_category_stats(category, year)
        price = max(1, min(1000, round(np.random.normal(stats['price_mean'], stats['price_std']), 2)))

        if day_type == 'special':
            units_multiplier = np.random.uniform(1.5, 3.0)
            discount = random.choice([0, 10, 15, 20, 25, 30, 40, 50])
        elif day_type == 'weekend':
            units_multiplier = np.random.uniform(1.2, 1.8)
            discount = random.choice([0, 5, 10, 15, 20])
        else:
            units_multiplier = np.random.uniform(0.8, 1.2)
            discount = random.choice([0, 0, 5, 10])

        units = int(max(0, np.random.normal(stats['units_mean'], stats['units_std']) * units_multiplier))
        competitor_pricing = max(0.5, round(price * (1 + np.random.uniform(-0.2, 0.15)), 2))

        weather_weights = {
            'North': {'Winter': [0.10, 0.20, 0.20, 0.40, 0.05, 0.05], 'Spring': [0.30, 0.40, 0.20, 0.00, 0.05, 0.05], 'Summer': [0.40, 0.30, 0.20, 0.00, 0.05, 0.05], 'Autumn': [0.20, 0.30, 0.30, 0.10, 0.05, 0.05]},
            'South': {'Winter': [0.30, 0.30, 0.20, 0.10, 0.05, 0.05], 'Spring': [0.40, 0.30, 0.20, 0.00, 0.05, 0.05], 'Summer': [0.50, 0.20, 0.20, 0.00, 0.05, 0.05], 'Autumn': [0.30, 0.30, 0.20, 0.10, 0.05, 0.05]},
            'East': {'Winter': [0.20, 0.30, 0.20, 0.20, 0.05, 0.05], 'Spring': [0.35, 0.35, 0.20, 0.00, 0.05, 0.05], 'Summer': [0.40, 0.30, 0.20, 0.00, 0.05, 0.05], 'Autumn': [0.25, 0.30, 0.25, 0.15, 0.05, 0.05]},
            'West': {'Winter': [0.30, 0.30, 0.10, 0.20, 0.05, 0.05], 'Spring': [0.40, 0.30, 0.15, 0.00, 0.10, 0.05], 'Summer': [0.50, 0.20, 0.10, 0.00, 0.10, 0.10], 'Autumn': [0.35, 0.30, 0.20, 0.05, 0.05, 0.05]},
            'Central': {'Winter': [0.20, 0.40, 0.10, 0.20, 0.05, 0.05], 'Spring': [0.30, 0.40, 0.20, 0.00, 0.05, 0.05], 'Summer': [0.40, 0.30, 0.10, 0.00, 0.10, 0.10], 'Autumn': [0.20, 0.30, 0.30, 0.10, 0.05, 0.05]},
            'International': {'Winter': [0.40, 0.20, 0.20, 0.00, 0.10, 0.10], 'Spring': [0.40, 0.30, 0.20, 0.00, 0.05, 0.05], 'Summer': [0.50, 0.20, 0.10, 0.00, 0.10, 0.10], 'Autumn': [0.30, 0.30, 0.20, 0.10, 0.05, 0.05]}
        }
        region_key = region if region in weather_weights else 'Central'
        probabilities = weather_weights[region_key][season]
        normalized_probabilities = [p / sum(probabilities) for p in probabilities]
        weather = np.random.choice(WEATHER_CONDITIONS, p=normalized_probabilities)
        holiday_promotion = 1 if (day_type == 'special' or random.random() < 0.3) else 0

        data.append([date.strftime("%Y-%m-%d"), product, category, region, units, price, discount, weather, holiday_promotion, competitor_pricing, season])
    return data

def process_date_chunk(date_chunk, special_days):
    chunk_data = []
    for date in date_chunk:
        chunk_data.append(generate_day_data(date, special_days))
    return chunk_data

def generate_inventory_data_parallel(start_date_str, end_date_str, num_workers=4):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    all_special_days = set()
    for year in range(start_date.year, end_date.year + 1):
        all_special_days.update(get_special_days(year))

    date_list = []
    current_date = start_date
    while current_date <= end_date:
        date_list.append(current_date)
        current_date += timedelta(days=1)

    chunk_size = max(1, len(date_list) // num_workers)
    date_chunks = [date_list[i:i + chunk_size] for i in range(0, len(date_list), chunk_size)]
    all_data = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_date_chunk, chunk, all_special_days): chunk for chunk in date_chunks}
        for future in as_completed(futures):
            all_data.extend(future.result())

    flat_data = [item for sublist in all_data for item in sublist]
    df = pd.DataFrame(flat_data, columns=['Date', 'Product ID', 'Category', 'Region', 'Units Sold', 'Price', 'Discount', 'Weather Condition', 'Holiday/Promotion', 'Competitor Pricing', 'Seasonality'])
    return df

def feature_engineering(df):
    df['Date'] = pd.to_datetime(df['Date'])
    daily_sales_df = df.groupby('Date').agg(
        total_units_sold=('Units Sold', 'sum'),
        weather_condition=('Weather Condition', 'first'),
        seasonality=('Seasonality', 'first'),
        holiday_promotion=('Holiday/Promotion', 'max')
    ).reset_index()
    daily_sales_df.rename(columns={'total_units_sold': 'Total Units Sold'}, inplace=True)
    
    daily_sales_df['year'] = daily_sales_df['Date'].dt.year
    daily_sales_df['month'] = daily_sales_df['Date'].dt.month
    daily_sales_df['dayofweek'] = daily_sales_df['Date'].dt.dayofweek
    daily_sales_df['dayofyear'] = daily_sales_df['Date'].dt.dayofyear
    daily_sales_df['weekofyear'] = daily_sales_df['Date'].dt.isocalendar().week.astype(int)
    
    daily_sales_df['Total Units Sold_lag_7'] = daily_sales_df['Total Units Sold'].shift(7)
    daily_sales_df['Total Units Sold_roll_3_mean'] = daily_sales_df['Total Units Sold'].rolling(window=3).mean()
    daily_sales_df['Total Units Sold_roll_7_mean'] = daily_sales_df['Total Units Sold'].rolling(window=7).mean()
    
    daily_sales_df['Target_Units_Sold'] = daily_sales_df['Total Units Sold'].shift(-7)
    daily_sales_df.dropna(inplace=True)
    
    daily_sales_df = pd.get_dummies(daily_sales_df, columns=['weather_condition', 'seasonality'], drop_first=True, dtype=int)
    return daily_sales_df

def main():
    print("Génération des données...")
    num_workers = min(mp.cpu_count(), 4)
    raw_df = generate_inventory_data_parallel(START_DATE, END_DATE, num_workers)
    
    print("Ingénierie des caractéristiques...")
    processed_df = feature_engineering(raw_df)
    
    X = processed_df.drop(columns=['Date', 'Total Units Sold', 'Target_Units_Sold'])
    y = processed_df['Target_Units_Sold']
    
    split_index = int(len(processed_df) * 0.8)
    split_date = processed_df.iloc[split_index]['Date']
    
    X_train = X[processed_df['Date'] < split_date]
    X_test = X[processed_df['Date'] >= split_date]
    y_train = y[processed_df['Date'] < split_date]
    y_test = y[processed_df['Date'] >= split_date]
    
    output_dir = "donnees_pretraitees"
    os.makedirs(output_dir, exist_ok=True)
    
    X_train.to_csv(f"{output_dir}/X_train_7_jours.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test_7_jours.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train_7_jours.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test_7_jours.csv", index=False)
    
    print(f"Données sauvegardées dans {output_dir}/")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

if __name__ == "__main__":
    main()
