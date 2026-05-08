import time
import pandas as pd

t = time.time()
for _ in range(365):
    pd.read_csv('example-data/2025-01-01.taxi-rides.csv')
csv_time = time.time() - t

t = time.time()
for _ in range(365):
    pd.read_parquet('example-data/2025-01-01.taxi-rides.parquet')
pq_time = time.time() - t

print(f'CSV:     {csv_time:.1f}s')
print(f'Parquet: {pq_time:.2f}s')
print(f'Parquet is {csv_time / pq_time:.0f}x faster')