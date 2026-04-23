"""
Project: AI Engineer 2026 - Week 1
Author: Amila Dilshan
Date: April 23, 2026 (Day 02)

Architecture Principles Applied:
- Single Responsibility Principle
- Separation of Concerns
- Clean, readable, typed code
- Proper logging and error handling
"""

import logging
from typing import List, Generator, Callable
import time

# ------------------- Standalone Timer Decorator (Fixed) -------------------
def timer_decorator(func: Callable):
    """Decorator that measures how long a function takes."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logging.info(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

# ------------------- Logging Setup -------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ------------------- DataPipeline Class -------------------
class DataPipeline:
    """Professional DataPipeline showing advanced Python features."""

    def __init__(self, name: str):
        self.name = name
        logging.info(f"Pipeline '{name}' initialized")

    # Context manager
    def __enter__(self):
        logging.info("Pipeline started")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.info("Pipeline finished")
        return False

    # Generator for batches
    def batch_generator(self, data: List[int], batch_size: int) -> Generator:
        """Yields data in small batches."""
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]

    # Example method using the decorator + context manager + generator
    @timer_decorator
    def process_data(self, data: List[int]) -> List[int]:
        """Process data using generator and context manager."""
        with self:   # using context manager
            processed = []
            for batch in self.batch_generator(data, batch_size=100):
                processed.extend([x * 2 for x in batch])
            return processed


# ------------------- Quick Test -------------------
if __name__ == "__main__":
    pipeline = DataPipeline("MyFirstPipeline")
    sample_data = list(range(1000))
    result = pipeline.process_data(sample_data)
    print(f"✅ Processed {len(result)} items successfully!")
    print(f"First 10 results: {result[:10]}")