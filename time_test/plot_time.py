import numpy as np

energy_gpu = np.genfromtxt("test1.txt")
class_gpu = np.genfromtxt("test2.txt")
zenith_gpu = np.genfromtxt("test3.txt")
total_gpu =  np.genfromtxt("total.txt")
energy_cpu = np.genfromtxt("test1_cpu.txt")
class_cpu = np.genfromtxt("test2_cpu.txt")
zenith_cpu = np.genfromtxt("test3_cpu.txt")
total_cpu =  np.genfromtxt("total_cpu.txt")
num_events =  np.genfromtxt("events.txt")

gpu_ml = energy_gpu + class_gpu + zenith_gpu
cpu_ml = energy_cpu + class_cpu + zenith_cpu
gpu_ml_per_event = gpu_ml / num_events
cpu_ml_per_event = cpu_ml / num_events

print("Average time for GPU per event: %f seconds"%np.mean(gpu_ml_per_event))
print("Average time for CPU per event: %f seconds"%np.mean(cpu_ml_per_event))
