"""
Enhanced performance monitoring system with improved metrics tracking
"""
import logging
import time
from typing import Dict, Optional
from collections import defaultdict
from functools import wraps
import numpy as np

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    def __init__(self):
        self.timers = {}
        self.history = defaultdict(list)
        self.slow_threshold = 1.0  # seconds
        self.metrics = defaultdict(lambda: {
            'count': 0,
            'total_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'times': []
        })
        
    def start_timer(self, category: str, name: str) -> str:
        """Start a new performance timer"""
        timer_id = f"{category}_{name}_{time.time()}"
        self.timers[timer_id] = {
            'category': category,
            'name': name,
            'start': time.perf_counter(),
            'checkpoints': []
        }
        return timer_id
        
    def add_checkpoint(self, timer_id: str, checkpoint_name: str) -> None:
        """Add a checkpoint to track intermediate timing"""
        if timer_id in self.timers:
            checkpoint_time = time.perf_counter()
            self.timers[timer_id]['checkpoints'].append({
                'name': checkpoint_name,
                'time': checkpoint_time
            })
            
    def stop_timer(self, timer_id: str, note: Optional[str] = None) -> float:
        """Stop timer and record metrics"""
        if timer_id not in self.timers:
            return 0.0
            
        end_time = time.perf_counter()
        timer = self.timers[timer_id]
        duration = end_time - timer['start']
        
        # Record metrics
        category = timer['category']
        name = timer['name']
        metric_key = f"{category}_{name}"
        
        self.metrics[metric_key]['count'] += 1
        self.metrics[metric_key]['total_time'] += duration
        self.metrics[metric_key]['min_time'] = min(
            self.metrics[metric_key]['min_time'], 
            duration
        )
        self.metrics[metric_key]['max_time'] = max(
            self.metrics[metric_key]['max_time'], 
            duration
        )
        self.metrics[metric_key]['times'].append(duration)
        
        # Keep history for trend analysis
        self.history[metric_key].append({
            'duration': duration,
            'timestamp': time.time(),
            'note': note
        })
        
        # Log if operation was slow
        if duration > self.slow_threshold:
            checkpoints = timer.get('checkpoints', [])
            if checkpoints:
                checkpoint_info = "\n".join(
                    f"  - {cp['name']}: {cp['time'] - timer['start']:.3f}s"
                    for cp in checkpoints
                )
                logger.warning(
                    f"Slow operation detected: {metric_key} took {duration:.3f}s\n"
                    f"Checkpoints:\n{checkpoint_info}"
                )
            else:
                logger.warning(
                    f"Slow operation detected: {metric_key} took {duration:.3f}s"
                )
                
        # Cleanup
        del self.timers[timer_id]
        return duration
        
    def get_metrics(self, category: Optional[str] = None) -> Dict:
        """Get performance metrics with optional category filter"""
        metrics = {}
        for key, data in self.metrics.items():
            if category and not key.startswith(category):
                continue
                
            times = data['times']
            if not times:
                continue
                
            metrics[key] = {
                'count': data['count'],
                'total_time': data['total_time'],
                'avg_time': data['total_time'] / data['count'],
                'min_time': data['min_time'],
                'max_time': data['max_time'],
                'median_time': np.median(times),
                'p95_time': np.percentile(times, 95) if len(times) > 1 else times[0]
            }
            
        return metrics
        
    def get_trends(self, metric_key: str, window: int = 10) -> Dict:
        """Analyze performance trends for a specific metric"""
        history = self.history[metric_key]
        if not history:
            return {}
            
        times = [h['duration'] for h in history[-window:]]
        if not times:
            return {}
            
        return {
            'current': times[-1],
            'trend': (
                'improving' if times[-1] < np.mean(times[:-1]) 
                else 'degrading' if times[-1] > np.mean(times[:-1])
                else 'stable'
            ),
            'avg_time': np.mean(times),
            'trend_slope': np.polyfit(range(len(times)), times, 1)[0]
        }
        
    def reset(self) -> None:
        """Reset all metrics"""
        self.timers.clear()
        self.metrics.clear()
        self.history.clear()
        
    def get_bottlenecks(self, threshold_pct: float = 10.0) -> Dict:
        """Identify performance bottlenecks"""
        metrics = self.get_metrics()
        total_time = sum(m['total_time'] for m in metrics.values())
        
        if not total_time:
            return {}
            
        bottlenecks = {}
        for key, data in metrics.items():
            time_pct = (data['total_time'] / total_time) * 100
            if time_pct >= threshold_pct:
                bottlenecks[key] = {
                    **data,
                    'time_percentage': time_pct
                }
                
        return bottlenecks
        
    def suggest_optimizations(self) -> Dict:
        """Suggest potential optimizations based on metrics"""
        suggestions = {}
        bottlenecks = self.get_bottlenecks()
        
        for key, data in bottlenecks.items():
            category = key.split('_')[0]
            
            if category == 'visualization':
                if data['avg_time'] > 0.1:  # 100ms threshold for visualization
                    suggestions[key] = [
                        "Consider reducing plot complexity",
                        "Enable downsampling for large datasets",
                        "Disable antialiasing for faster rendering",
                        "Increase batch size for plotting"
                    ]
            elif category == 'data_processing':
                if data['avg_time'] > 0.5:  # 500ms threshold for data processing
                    suggestions[key] = [
                        "Use vectorized operations",
                        "Implement data caching",
                        "Optimize data types",
                        "Process data in chunks"
                    ]
                    
        return suggestions

    def print_report(self) -> None:
        """Print a formatted performance report to the console"""
        metrics = self.get_metrics()
        if not metrics:
            print("No performance metrics recorded.")
            return

        print("\n=== Performance Report ===")
        print(f"Total operations tracked: {sum(m['count'] for m in metrics.values())}")
        
        # Print metrics by category
        categories = set(key.split('_')[0] for key in metrics.keys())
        for category in sorted(categories):
            print(f"\n{category.upper()}:")
            category_metrics = {k: v for k, v in metrics.items() if k.startswith(category)}
            
            for key, data in sorted(category_metrics.items()):
                operation = key.replace(f"{category}_", "")
                print(f"\n  {operation}:")
                print(f"    Count: {data['count']}")
                print(f"    Average time: {data['avg_time']:.3f}s")
                print(f"    Min time: {data['min_time']:.3f}s")
                print(f"    Max time: {data['max_time']:.3f}s")
                print(f"    P95 time: {data['p95_time']:.3f}s")

        # Print bottlenecks if any
        bottlenecks = self.get_bottlenecks()
        if bottlenecks:
            print("\nPotential Bottlenecks:")
            for key, data in bottlenecks.items():
                print(f"  {key}: {data['time_percentage']:.1f}% of total time")

        # Print optimization suggestions
        suggestions = self.suggest_optimizations()
        if suggestions:
            print("\nOptimization Suggestions:")
            for key, tips in suggestions.items():
                print(f"\n  {key}:")
                for tip in tips:
                    print(f"    - {tip}")
        print("\n=========================")

def function_timer(category: str):
    """Decorator to time function execution"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create monitor instance, checking if it exists on the instance or using global
            monitor = (args[0].perf_monitor if args and hasattr(args[0], 'perf_monitor')
                      else globals().get('perf_monitor', PerformanceMonitor()))
                
            timer_id = monitor.start_timer(category, func.__name__)
            try:
                result = func(*args, **kwargs)
                monitor.stop_timer(timer_id)
                return result
            except Exception as e:
                monitor.stop_timer(timer_id, f"Error: {str(e)}")
                raise
        return wrapper
    return decorator