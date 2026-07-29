import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

NUM_ROWS = 100000
random.seed(42)
np.random.seed(42)

crop_types = [
    "Wheat",
    "Rice",
    "Corn",
    "Barley",
    "Soybean",
    "Potato",
    "Tomato",
    "Onion"
]

vehicle_types = [
    "Truck",
    "Van",
    "Refrigerated Truck",
    "Mini Truck"
]

start_date = datetime(2023, 1, 1)

rows = []

def clamp(value, low, high):
    return max(low, min(high, value))


def generate_record():

    crop = random.choice(crop_types)
    vehicle = random.choice(vehicle_types)

    harvest_date = (
        start_date +
        timedelta(days=random.randint(0, 730))
    ).strftime("%Y-%m-%d")

    crop_yield = round(random.uniform(2.5, 8.5), 2)

    storage_temperature = round(random.uniform(2, 35), 2)
    storage_humidity = round(random.uniform(40, 95), 2)

    fuel_consumption = round(random.uniform(8, 120), 2)

    route_distance = round(random.uniform(20, 1200), 2)

    delivery_time = round(
        route_distance / random.uniform(35, 70),
        2
    )

    traffic_level = round(random.uniform(0, 10), 2)

    temperature = round(random.uniform(5, 45), 2)

    humidity = round(random.uniform(35, 95), 2)

    vehicle_load_capacity = round(random.uniform(200, 12000), 2)

    vibration_level = round(random.uniform(0, 350), 2)

    queue_time = round(random.uniform(0, 10), 2)

    weather_impact = round(random.uniform(0, 10), 2)

    station_capacity = round(random.uniform(500, 10000), 2)

    operational_cost = round(random.uniform(500, 10000), 2)

    energy_consumption = round(random.uniform(50, 1200), 2)

    iot_temperature = round(
        storage_temperature + random.uniform(-2, 2),
        2
    )

    iot_humidity = round(
        storage_humidity + random.uniform(-5, 5),
        2
    )

    iot_light = round(random.uniform(0, 100), 2)

    warehouse_storage_time = round(
        random.uniform(0, 30),
        2
    )

    inventory_levels = random.randint(50, 5000)

    fuel_costs = round(
        fuel_consumption * random.uniform(1.2, 2.2),
        2
    )

    efficiency_ratio = round(
        random.uniform(60, 100),
        2
    )

    quality_maintenance_ratio = round(
        random.uniform(40, 100),
        2
    )

    # ==========================================
    # Continuous Spoilage Score (0–1)
    # ==========================================

    spoilage_score = (
        0.18 * (storage_temperature / 35)
        + 0.15 * (storage_humidity / 100)
        + 0.15 * (warehouse_storage_time / 30)
        + 0.10 * (delivery_time / 35)
        + 0.08 * (traffic_level / 10)
        + 0.08 * (weather_impact / 10)
        + 0.08 * (vibration_level / 350)
        + 0.06 * (route_distance / 1200)
        + 0.05 * (fuel_consumption / 120)
        - 0.12 * (quality_maintenance_ratio / 100)
        - 0.05 * (efficiency_ratio / 100)
    )

    spoilage_score += random.uniform(-0.03, 0.03)

    spoilage_score = clamp(
        spoilage_score,
        0.0,
        1.0
    )
        # ==========================================
    # Numeric Target for ML
    # ==========================================

    spoilage_risk = round(spoilage_score * 100, 2)

    return [
        vehicle,
        crop,
        "2025-01-01",
        crop_yield,
        storage_temperature,
        storage_humidity,
        fuel_consumption,
        route_distance,
        delivery_time,
        traffic_level,
        temperature,
        humidity,
        vehicle_load_capacity,
        vibration_level,
        queue_time,
        weather_impact,
        station_capacity,
        operational_cost,
        energy_consumption,
        iot_temperature,
        iot_humidity,
        iot_light,
        warehouse_storage_time,
        inventory_levels,
        fuel_costs,
        spoilage_risk,
        efficiency_ratio,
        quality_maintenance_ratio
    ]

    # ==========================================
# Generate Dataset
# ==========================================

for _ in range(NUM_ROWS):
    rows.append(generate_record())

columns = [
    "Vehicle_Type",
    "Crop_Type",
    "Harvest_Date",
    "Crop_Yield",
    "Storage_Temperature",
    "Storage_Humidity",
    "Fuel_Consumption",
    "Route_Distance",
    "Delivery_Time",
    "Traffic_Level",
    "Temperature",
    "Humidity",
    "Vehicle_Load_Capacity",
    "Vibration_Level",
    "Queue_Time",
    "Weather_Impact",
    "Station_Capacity",
    "Operational_Cost",
    "Energy_Consumption",
    "IoT_Sensor_Reading_Temperature",
    "IoT_Sensor_Reading_Humidity",
    "IoT_Sensor_Reading_Light",
    "Warehouse_Storage_Time",
    "Inventory_Levels",
    "Fuel_Costs",
    "Spoilage_Risk",
    "Efficiency_Ratio",
    "Quality_Maintenance_Ratio"
]

df = pd.DataFrame(rows, columns=columns)

# ==========================================
# Save Dataset
# ==========================================

output_file = "data/food_loss_dataset_100k.csv"

df.to_csv(output_file, index=False)

print("=" * 60)
print("Dataset generation completed successfully!")
print("=" * 60)
print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")
print(f"Saved to: {output_file}")
print("=" * 60)

print("\nFirst 5 rows:\n")
print(df.head())

print("\nSpoilage Risk Statistics:")
print(df["Spoilage_Risk"].describe())

print("\nMissing Values:")
print(df.isnull().sum())

# ==========================================
# Save Dataset Statistics
# ==========================================

stats = pd.DataFrame({
    "Feature": df.columns,
    "Missing_Values": df.isnull().sum().values,
    "Data_Type": df.dtypes.astype(str).values
})

stats.to_csv("data/dataset_statistics.csv", index=False)

print("\nDataset Statistics saved to:")
print("data/dataset_statistics.csv")

print("\nSpoilage Risk Range:")
print(f"Min : {df['Spoilage_Risk'].min():.2f}")
print(f"Max : {df['Spoilage_Risk'].max():.2f}")
print(f"Mean: {df['Spoilage_Risk'].mean():.2f}")

print("\nDataset generation completed successfully.")

