

"""
Generate Training Dataset for AIoT Workshop
Creates 100 bus trajectories (50 each direction) with realistic variations

if you want to reain your model with even more datasets, here is an example of a script that generates datasets, good luck :)

"""
import requests
import csv
import random
import json
from datetime import datetime, timedelta

# Route endpoints
FACULTY_LAT, FACULTY_LON = 30.3777, -9.5338
TAGHAZOUT_LAT, TAGHAZOUT_LON = 30.5428, -9.7089

def fetch_route(start_lat, start_lon, end_lat, end_lon):
    """Fetch route from OSRM"""
    url = f"http://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}"
    params = {'overview': 'full', 'geometries': 'geojson'}
    
    try:
        response = requests.get(url, params=params, timeout=15)
        data = response.json()
        
        if data['code'] == 'Ok':
            coords = data['routes'][0]['geometry']['coordinates']
            return [[lat, lon] for lon, lat in coords]
        return None
    except Exception as e:
        print(f"Error fetching route: {e}")
        return None

def add_realistic_variations(waypoints, trajectory_id):
    """Add realistic variations to make each trajectory unique"""
    
    # Time of day affects speed and passenger count
    hour = (6 + (trajectory_id % 18))  # 6am to 11pm
    is_rush_hour = (7 <= hour <= 9) or (17 <= hour <= 19)
    is_weekend = (trajectory_id % 7) in [5, 6]
    
    # Weather conditions
    weather_conditions = ['sunny', 'cloudy', 'rainy', 'windy']
    weather = random.choice(weather_conditions)
    
    # Traffic level
    if is_rush_hour and not is_weekend:
        traffic_level = random.choice(['heavy', 'heavy', 'moderate'])
    elif is_weekend:
        traffic_level = random.choice(['light', 'moderate'])
    else:
        traffic_level = random.choice(['light', 'moderate', 'moderate'])
    
    # Initial passenger count
    if is_rush_hour:
        initial_passengers = random.randint(25, 35)
    else:
        initial_passengers = random.randint(10, 25)
    
    # Speed variations based on conditions
    speed_multiplier = 1.0
    if weather == 'rainy':
        speed_multiplier *= 0.8
    if traffic_level == 'heavy':
        speed_multiplier *= 0.7
    elif traffic_level == 'moderate':
        speed_multiplier *= 0.85
    
    return {
        'hour': hour,
        'is_rush_hour': is_rush_hour,
        'is_weekend': is_weekend,
        'weather': weather,
        'traffic_level': traffic_level,
        'initial_passengers': initial_passengers,
        'speed_multiplier': speed_multiplier
    }

def generate_trajectory(waypoints, trajectory_id, direction, context):
    """Generate one complete trajectory with all data points"""
    
    total_distance = 31.6  # km
    data_points = []
    
    # Passenger dynamics
    passengers = context['initial_passengers']
    
    for i, (lat, lon) in enumerate(waypoints):
        progress = i / len(waypoints)
        distance_covered = progress * total_distance
        distance_remaining = total_distance - distance_covered
        
        # Calculate speed (with variations)
        if distance_covered < 10:  # Urban
            base_speed = random.randint(25, 35)
        else:  # Highway
            base_speed = random.randint(50, 65)
        
        speed = int(base_speed * context['speed_multiplier'])
        
        # Passenger changes at stops (every ~2km)
        if i > 0 and i % 50 == 0:
            if direction == 'to_taghazout':
                # Morning: people going to work/beach
                if context['hour'] < 12:
                    passengers += random.randint(-2, 5)
                else:
                    passengers += random.randint(-5, 2)
            else:
                # Evening: people returning
                if context['hour'] > 17:
                    passengers += random.randint(-2, 5)
                else:
                    passengers += random.randint(-5, 2)
            
            passengers = max(0, min(passengers, 40))  # Capacity 0-40
        
        # Create data point
        data_point = {
            'trajectory_id': trajectory_id,
            'direction': direction,
            'point_index': i,
            'timestamp': (datetime.now() + timedelta(seconds=i)).isoformat(),
            'lat': round(lat, 6),
            'lon': round(lon, 6),
            'speed_kmh': speed,
            'distance_covered_km': round(distance_covered, 2),
            'distance_remaining_km': round(distance_remaining, 2),
            'progress_percent': round(progress * 100, 1),
            'passengers': passengers,
            'hour_of_day': context['hour'],
            'is_rush_hour': int(context['is_rush_hour']),
            'is_weekend': int(context['is_weekend']),
            'weather': context['weather'],
            'traffic_level': context['traffic_level']
        }
        
        data_points.append(data_point)
    
    return data_points

def main():
    print("=" * 70)
    print("AIOT WORKSHOP - TRAINING DATASET GENERATOR")
    print("=" * 70)
    print()
    
    # Fetch both routes
    print("📍 Fetching routes from OSRM...")
    route_to_taghazout = fetch_route(FACULTY_LAT, FACULTY_LON, TAGHAZOUT_LAT, TAGHAZOUT_LON)
    route_to_faculty = fetch_route(TAGHAZOUT_LAT, TAGHAZOUT_LON, FACULTY_LAT, FACULTY_LON)
    
    if not route_to_taghazout or not route_to_faculty:
        print("❌ Failed to fetch routes")
        return
    
    print(f"✅ Route Faculty→Taghazout: {len(route_to_taghazout)} points")
    print(f"✅ Route Taghazout→Faculty: {len(route_to_faculty)} points")
    print()
    
    # Generate dataset
    print("🔄 Generating 100 trajectories...")
    print()
    
    all_data = []
    
    # 50 trajectories Faculty → Taghazout
    for i in range(50):
        context = add_realistic_variations(route_to_taghazout, i)
        trajectory_data = generate_trajectory(
            route_to_taghazout, 
            f"T{i:03d}", 
            'to_taghazout',
            context
        )
        all_data.extend(trajectory_data)
        print(f"✅ Generated trajectory T{i:03d} (Faculty→Taghazout) - {len(trajectory_data)} points")
    
    # 50 trajectories Taghazout → Faculty
    for i in range(50, 100):
        context = add_realistic_variations(route_to_faculty, i)
        trajectory_data = generate_trajectory(
            route_to_faculty,
            f"T{i:03d}",
            'to_faculty',
            context
        )
        all_data.extend(trajectory_data)
        print(f"✅ Generated trajectory T{i:03d} (Taghazout→Faculty) - {len(trajectory_data)} points")
    
    # Save to CSV
    output_file = 'training_dataset.csv'
    
    fieldnames = [
        'trajectory_id', 'direction', 'point_index', 'timestamp',
        'lat', 'lon', 'speed_kmh', 
        'distance_covered_km', 'distance_remaining_km', 'progress_percent',
        'passengers', 'hour_of_day', 'is_rush_hour', 'is_weekend',
        'weather', 'traffic_level'
    ]
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_data)
    
    print()
    print("=" * 70)
    print("✅ DATASET GENERATION COMPLETE!")
    print("=" * 70)
    print(f"📊 Total data points: {len(all_data):,}")
    print(f"💾 Saved to: {output_file}")
    print(f"📏 File size: {len(all_data) * len(fieldnames)} values")
    print()
    print("Dataset features:")
    print("  - 100 trajectories (50 each direction)")
    print("  - GPS coordinates (lat, lon)")
    print("  - Speed variations")
    print("  - Passenger count dynamics")
    print("  - Time of day (hour, rush hour, weekend)")
    print("  - Weather conditions")
    print("  - Traffic levels")
    print()
    print("Students can use this to train models for:")
    print("  ✓ Predicting passenger count")
    print("  ✓ Estimating arrival time")
    print("  ✓ Speed prediction")
    print("  ✓ Route optimization")
    print("=" * 70)

if __name__ == "__main__":
    main()
