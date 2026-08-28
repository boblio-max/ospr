import time
import math

def bI(x):
    c_pi = 3.141592653589793
    return ((16 * x) * (c_pi - x) / (5 * (c_pi * c_pi)) - 4 * x * (c_pi - 4)) + 0.00001343 * x * (c_pi - x) * (x - c_pi / 2) ** 2
times = []

fstart = time.perf_counter()
for i in range(40):
    start = time.perf_counter()
    print(bI(i * 0.1))
    end = time.perf_counter()
    times.append(end - start)
    
fend = time.perf_counter()

print(f"total times: {fend - fstart}")
print(f"individual times: {times}")

times = [] 

fstart = time.perf_counter()
for i in range(40):
    start = time.perf_counter()
    print(math.sin(i * 0.1))
    end = time.perf_counter()
    times.append(end - start)
    
fend = time.perf_counter()

print(f"total times: {fend - fstart}")
print(f"individual times: {times}")