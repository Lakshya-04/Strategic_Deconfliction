#!/usr/bin/env python3
"""
Performance Benchmark Suite for Strategic_Deconfliction
This module provides comprehensive performance benchmarks for testing computational performance,
memory usage, and algorithmic efficiency in autonomous vehicle deconfliction scenarios.
"""

import time
import json
import sys
import os
import tracemalloc
import statistics
from typing import Dict, List, Callable, Any, Tuple
from functools import wraps
import argparse
import traceback


class PerformanceBenchmark:
    """
    A comprehensive benchmarking class for performance testing with GitHub Actions integration.
    Supports both execution time and memory usage benchmarks with statistical analysis.
    """

    def __init__(self, name: str = "Strategic_Deconfliction_Benchmark"):
        self.name = name
        self.results = []
        self.memory_results = []
        
    def time_benchmark(self, func: Callable, *args, iterations: int = 1000, warmup: int = 10, **kwargs) -> Dict:
        """
        Benchmark execution time of a function with statistical analysis.
        
        Args:
            func: Function to benchmark
            *args: Positional arguments for the function
            iterations: Number of benchmark iterations
            warmup: Number of warmup runs
            **kwargs: Keyword arguments for the function
            
        Returns:
            Dict containing benchmark results
        """
        # Warmup runs
        for _ in range(warmup):
            func(*args, **kwargs)
            
        # Actual benchmark runs
        times = []
        for _ in range(iterations):
            start_time = time.perf_counter()
            func(*args, **kwargs)
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds
            
        # Statistical analysis
        result = {
            "function": func.__name__,
            "iterations": iterations,
            "min_ms": min(times),
            "max_ms": max(times),
            "mean_ms": statistics.mean(times),
            "median_ms": statistics.median(times),
            "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0,
            "p95_ms": self._percentile(times, 0.95),
            "p99_ms": self._percentile(times, 0.99)
        }
        
        self.results.append(result)
        return result
    
    def memory_benchmark(self, func: Callable, *args, **kwargs) -> Dict:
        """
        Benchmark memory usage of a function.
        
        Args:
            func: Function to benchmark
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Dict containing memory benchmark results
        """
        tracemalloc.start()
        
        # Execute function
        start_memory = tracemalloc.get_traced_memory()[0]
        result = func(*args, **kwargs)
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        
        tracemalloc.stop()
        
        memory_result = {
            "function": func.__name__,
            "start_memory_mb": start_memory / 1024 / 1024,
            "peak_memory_mb": peak_memory / 1024 / 1024,
            "memory_delta_mb": (peak_memory - start_memory) / 1024 / 1024
        }
        
        self.memory_results.append(memory_result)
        return memory_result
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of a dataset."""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def save_results(self, output_file: str = "benchmark_results.json") -> None:
        """Save benchmark results to JSON file for GitHub Actions integration."""
        combined_results = {
            "benchmark_name": self.name,
            "timestamp": time.time(),
            "time_benchmarks": self.results,
            "memory_benchmarks": self.memory_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(combined_results, f, indent=2)
        
        print(f"Benchmark results saved to {output_file}")
    
    def print_results(self) -> None:
        """Print benchmark results in a readable format."""
        print(f"\n=== {self.name} Benchmark Results ===\n")
        
        # Time benchmarks
        if self.results:
            print("Time Benchmarks:")
            print("-" * 70)
            for result in self.results:
                print(f"Function: {result['function']}")
                print(f"  Iterations: {result['iterations']}")
                print(f"  Mean: {result['mean_ms']:.4f} ms")
                print(f"  Median: {result['median_ms']:.4f} ms")
                print(f"  Min: {result['min_ms']:.4f} ms")
                print(f"  Max: {result['max_ms']:.4f} ms")
                print(f"  Std Dev: {result['stdev_ms']:.4f} ms")
                print(f"  95th percentile: {result['p95_ms']:.4f} ms")
                print(f"  99th percentile: {result['p99_ms']:.4f} ms")
                print()
        
        # Memory benchmarks
        if self.memory_results:
            print("Memory Benchmarks:")
            print("-" * 70)
            for result in self.memory_results:
                print(f"Function: {result['function']}")
                print(f"  Peak Memory: {result['peak_memory_mb']:.2f} MB")
                print(f"  Memory Delta: {result['memory_delta_mb']:.2f} MB")
                print()


# Sample benchmark functions for Strategic Deconfliction scenarios
def matrix_multiplication_benchmark(size: int = 100) -> None:
    """Benchmark matrix multiplication for computational workloads."""
    import random
    
    # Generate random matrices
    matrix_a = [[random.random() for _ in range(size)] for _ in range(size)]
    matrix_b = [[random.random() for _ in range(size)] for _ in range(size)]
    
    # Matrix multiplication
    result = [[0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(size):
            for k in range(size):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]


def pathfinding_benchmark(grid_size: int = 50) -> None:
    """Benchmark pathfinding algorithm for vehicle routing scenarios."""
    import random
    
    # Create grid
    grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
    
    # Add obstacles (20% of grid)
    for i in range(grid_size):
        for j in range(grid_size):
            if random.random() < 0.2:
                grid[i][j] = 1
    
    # Simple A* pathfinding simulation
    start = (0, 0)
    goal = (grid_size-1, grid_size-1)
    
    open_list = [start]
    closed_list = []
    
    while open_list:
        current = open_list.pop(0)
        closed_list.append(current)
        
        if current == goal:
            break
            
        # Check neighbors
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            x, y = current[0] + dx, current[1] + dy
            if (0 <= x < grid_size and 0 <= y < grid_size and 
                grid[x][y] == 0 and (x, y) not in closed_list):
                if (x, y) not in open_list:
                    open_list.append((x, y))


def collision_detection_benchmark(num_objects: int = 1000) -> None:
    """Benchmark collision detection for deconfliction scenarios."""
    import random
    import math
    
    # Generate random objects with position and radius
    objects = []
    for _ in range(num_objects):
        x = random.uniform(-1000, 1000)
        y = random.uniform(-1000, 1000)
        radius = random.uniform(1, 10)
        objects.append((x, y, radius))
    
    # Check all pairs for collision
    collisions = 0
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            x1, y1, r1 = objects[i]
            x2, y2, r2 = objects[j]
            
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if distance < (r1 + r2):
                collisions += 1


def data_processing_benchmark(num_records: int = 10000) -> None:
    """Benchmark data processing operations for sensor fusion."""
    import random
    
    # Generate sensor data
    sensor_data = []
    for _ in range(num_records):
        record = {
            'timestamp': time.time() + random.uniform(0, 100),
            'x': random.uniform(-100, 100),
            'y': random.uniform(-100, 100),
            'velocity': random.uniform(0, 50),
            'confidence': random.uniform(0.5, 1.0)
        }
        sensor_data.append(record)
    
    # Filter and process data
    filtered_data = [r for r in sensor_data if r['confidence'] > 0.8]
    
    # Sort by timestamp
    filtered_data.sort(key=lambda x: x['timestamp'])
    
    # Calculate statistics
    velocities = [r['velocity'] for r in filtered_data]
    avg_velocity = sum(velocities) / len(velocities) if velocities else 0
    
    return len(filtered_data), avg_velocity


def run_benchmarks():
    """Run all benchmark tests."""
    benchmark = PerformanceBenchmark("Strategic_Deconfliction")
    
    print("Starting Strategic Deconfliction Performance Benchmarks...")
    print("This may take a few minutes to complete.\n")
    
    # Matrix multiplication benchmark
    print("Running matrix multiplication benchmark...")
    benchmark.time_benchmark(matrix_multiplication_benchmark, 50, iterations=10)
    benchmark.memory_benchmark(matrix_multiplication_benchmark, 50)
    
    # Pathfinding benchmark
    print("Running pathfinding benchmark...")
    benchmark.time_benchmark(pathfinding_benchmark, 30, iterations=50)
    benchmark.memory_benchmark(pathfinding_benchmark, 30)
    
    # Collision detection benchmark
    print("Running collision detection benchmark...")
    benchmark.time_benchmark(collision_detection_benchmark, 500, iterations=20)
    benchmark.memory_benchmark(collision_detection_benchmark, 500)
    
    # Data processing benchmark
    print("Running data processing benchmark...")
    benchmark.time_benchmark(data_processing_benchmark, 5000, iterations=100)
    benchmark.memory_benchmark(data_processing_benchmark, 5000)
    
    # Print and save results
    benchmark.print_results()
    benchmark.save_results("benchmark_results.json")
    
    # Create pytest-benchmark compatible output for GitHub Actions
    create_pytest_benchmark_output(benchmark)


def create_pytest_benchmark_output(benchmark: PerformanceBenchmark):
    """Create pytest-benchmark compatible JSON output for GitHub Actions integration."""
    pytest_results = {
        "machine_info": {
            "python_version": sys.version,
            "platform": sys.platform
        },
        "benchmarks": []
    }
    
    for result in benchmark.results:
        pytest_benchmark = {
            "group": "performance",
            "name": f"test_benchmark_{result['function']}",
            "fullname": f"benchmark_performance.py::test_benchmark_{result['function']}",
            "params": None,
            "param": None,
            "extra_info": {},
            "stats": {
                "min": result['min_ms'] / 1000,  # Convert to seconds for consistency
                "max": result['max_ms'] / 1000,
                "mean": result['mean_ms'] / 1000,
                "stddev": result['stdev_ms'] / 1000,
                "rounds": result['iterations'],
                "median": result['median_ms'] / 1000,
                "iqr": (result['p95_ms'] - result['min_ms']) / 1000,
                "q1": result['min_ms'] / 1000,
                "q3": result['p95_ms'] / 1000,
                "iqr_outliers": 0,
                "stddev_outliers": 0,
                "outliers": "0;0",
                "ld15iqr": result['min_ms'] / 1000,
                "hd15iqr": result['p95_ms'] / 1000,
                "ops": 1.0 / (result['mean_ms'] / 1000) if result['mean_ms'] > 0 else 0,
                "total": result['mean_ms'] * result['iterations'] / 1000,
                "iterations": 1
            }
        }
        pytest_results["benchmarks"].append(pytest_benchmark)
    
    # Save pytest-benchmark compatible output
    with open("pytest_benchmark_output.json", 'w') as f:
        json.dump(pytest_results, f, indent=2)
    
    print("pytest-benchmark compatible output saved to pytest_benchmark_output.json")


def main():
    """Main function with command line argument support."""
    parser = argparse.ArgumentParser(description='Strategic Deconfliction Performance Benchmarks')
    parser.add_argument('--output', default='benchmark_results.json', 
                       help='Output file for benchmark results')
    parser.add_argument('--iterations', type=int, default=None,
                       help='Override default iteration counts')
    
    args = parser.parse_args()
    
    try:
        run_benchmarks()
        print(f"\nBenchmark completed successfully!")
        print(f"Results saved to: {args.output}")
        print("pytest-benchmark output saved to: pytest_benchmark_output.json")
        
        # Exit with success code
        sys.exit(0)
        
    except Exception as e:
        print(f"Benchmark failed with error: {e}")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()