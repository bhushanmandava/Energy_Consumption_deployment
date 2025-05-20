import requests
import numpy as np
import time
import random
from datetime import datetime, timedelta

# Base URL for the Energy Consumption Prediction API
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint of the API"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Health check status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print(f"Connection Error: Could not connect to the server at {BASE_URL}")
        return False

def generate_random_data():
    """Generate random data for energy consumption prediction"""
    # Random timestamp within the last 30 days
    random_days = random.randint(0, 30)
    random_hours = random.randint(0, 23)
    timestamp = (datetime.now() - timedelta(days=random_days, hours=random_hours)).strftime("%Y-%m-%d %H:%M:%S")
    
    # Generate random features based on realistic ranges
    data = {
        "Temperature": round(random.uniform(20, 30), 2),  # Temperature in degrees Celsius
        "Humidity": round(random.uniform(40, 60), 2),     # Humidity percentage
        "SquareFootage": round(random.uniform(1000, 2000), 2),  # Building square footage
        "Occupancy": random.randint(1, 10),               # Number of occupants
        "RenewableEnergy": round(random.uniform(1, 25), 2),  # Renewable energy percentage
        "Timestamp": timestamp,
        "HVACUsage": random.choice(["On", "Off"]),        # HVAC system status
        "LightingUsage": random.choice(["On", "Off"]),    # Lighting system status
        "Holiday": random.choice(["Yes", "No"])           # Holiday flag
    }
    
    return data

def send_prediction_request():
    """Send a single prediction request to the API"""
    data = generate_random_data()
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=data)
        print(f"Prediction request status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Prediction result: Energy Consumption = {result['energy_consumption']:.2f} units")
            print(f"For timestamp: {result['timestamp']}")
            print("\nInput parameters:")
            for key, value in data.items():
                print(f"  {key}: {value}")
        else:
            print(f"Error response: {response.text}")
        
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print(f"Connection Error: Could not connect to the server at {BASE_URL}")
        return False

def send_batch_prediction_request(batch_size=3):
    """Send a batch prediction request with multiple instances"""
    instances = [generate_random_data() for _ in range(batch_size)]
    batch_request = {"instances": instances}
    
    try:
        response = requests.post(f"{BASE_URL}/batch-predict", json=batch_request)
        print(f"Batch prediction request status: {response.status_code}")
        
        if response.status_code == 200:
            results = response.json()
            print(f"Received {len(results['predictions'])} predictions:")
            
            for i, prediction in enumerate(results['predictions']):
                print(f"\nPrediction {i+1}:")
                print(f"  Energy Consumption: {prediction['energy_consumption']:.2f} units")
                print(f"  Timestamp: {prediction['timestamp']}")
                print("  Input parameters:")
                for key, value in instances[i].items():
                    print(f"    {key}: {value}")
        else:
            print(f"Error response: {response.text}")
        
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print(f"Connection Error: Could not connect to the server at {BASE_URL}")
        return False

def simulate_user_traffic(num_requests=10, delay=1, include_batch=True):
    """Simulate user traffic with a mix of single and batch requests"""
    print(f"Simulating {num_requests} user requests with {delay} second delay between requests")
    
    success_count = 0
    total_predictions = 0
    
    for i in range(num_requests):
        print(f"\n--- Request {i+1}/{num_requests} ---")
        
        # Randomly choose between single prediction and batch prediction
        if not include_batch or random.random() < 0.7:  # 70% chance for single prediction if batches are enabled
            success = send_prediction_request()
            if success:
                success_count += 1
                total_predictions += 1
        else:
            # For batch predictions, randomly choose the batch size
            batch_size = random.randint(2, 5)
            success = send_batch_prediction_request(batch_size)
            if success:
                success_count += 1
                total_predictions += batch_size
        
        # Add delay between requests
        if i < num_requests - 1:
            time.sleep(delay)
    
    # Print summary
    print("\n--- Traffic Simulation Summary ---")
    print(f"Total requests sent: {num_requests}")
    print(f"Successful requests: {success_count}")
    print(f"Failed requests: {num_requests - success_count}")
    print(f"Total predictions made: {total_predictions}")
    print(f"Success rate: {(success_count/num_requests)*100:.1f}%")

def run_performance_test(num_requests=50, concurrent=False):
    """Run a performance test with many requests to measure response times"""
    print(f"Running performance test with {num_requests} requests...")
    
    start_time = time.time()
    response_times = []
    
    for i in range(num_requests):
        request_start = time.time()
        data = generate_random_data()
        
        try:
            response = requests.post(f"{BASE_URL}/predict", json=data)
            request_time = time.time() - request_start
            response_times.append(request_time)
            
            print(f"Request {i+1}/{num_requests}: Status {response.status_code}, Time {request_time:.4f}s")
            
            if not concurrent:
                # Small delay to prevent overwhelming the server
                time.sleep(0.1)
                
        except requests.exceptions.ConnectionError:
            print(f"Request {i+1}/{num_requests}: Connection failed")
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    if response_times:
        avg_time = sum(response_times) / len(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
        
        print("\n--- Performance Test Results ---")
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Average response time: {avg_time:.4f} seconds")
        print(f"Min response time: {min_time:.4f} seconds")
        print(f"Max response time: {max_time:.4f} seconds")
        print(f"Requests per second: {num_requests/total_time:.2f}")
    else:
        print("No successful responses recorded")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Client for Energy Consumption Prediction API")
    parser.add_argument("--health", action="store_true", help="Test health check endpoint")
    parser.add_argument("--predict", action="store_true", help="Send a single prediction request")
    parser.add_argument("--batch", action="store_true", help="Send a batch prediction request")
    parser.add_argument("--simulate", type=int, metavar="N", help="Simulate N user requests")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between simulated requests in seconds")
    parser.add_argument("--perf", type=int, metavar="N", help="Run a performance test with N requests")
    parser.add_argument("--concurrent", action="store_true", help="Don't add delay in performance tests")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="Base URL for the API")
    
    args = parser.parse_args()
    
    # Update the base URL if provided
    if args.url:
        BASE_URL = args.url
        print(f"Using API at: {BASE_URL}")
    
    # Default behavior if no options specified
    if not any(vars(args).values()):
        print("No options specified. Running health check and single prediction test.")
        test_health_check()
        print("\nSending a single prediction request:")
        send_prediction_request()
        print("\nSending a batch prediction request (3 instances):")
        send_batch_prediction_request(3)
    else:
        # Run the specified options
        if args.health:
            test_health_check()
        
        if args.predict:
            print("\nSending a single prediction request:")
            send_prediction_request()
        
        if args.batch:
            print("\nSending a batch prediction request (3 instances):")
            send_batch_prediction_request(3)
        
        if args.simulate:
            print("\nStarting user traffic simulation...")
            simulate_user_traffic(args.simulate, args.delay)
        
        if args.perf:
            print("\nRunning performance test...")
            run_performance_test(args.perf, args.concurrent)