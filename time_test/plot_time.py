import numpy as np
import matplotlib.pyplot as plt

save_folder="/mnt/home/micall12/LowEnergyNeuralNetwork/time_test/test_16July2021/"
basefile_name = "time_numu"
check_cpu = False

energy_gpu = np.genfromtxt("%senergy_%s.txt"%(save_folder,basefile_name))
class_gpu = np.genfromtxt("%sclass_%s.txt"%(save_folder,basefile_name))
zenith_gpu = np.genfromtxt("%szenith_%s.txt"%(save_folder,basefile_name))
vertex_gpu = np.genfromtxt("%svertex_%s.txt"%(save_folder,basefile_name))
muon_gpu = np.genfromtxt("%smuon_%s.txt"%(save_folder,basefile_name))
total_gpu =  np.genfromtxt("%stotal_%s.txt"%(save_folder,basefile_name))
num_events = np.genfromtxt("%snumber_events_numu.txt"%save_folder) 

gpu_ml = energy_gpu + class_gpu + zenith_gpu + vertex_gpu + muon_gpu
gpu_ml_per_event = gpu_ml / num_events
gpu_total_per_event = total_gpu / num_events
print("Average time for CNN on GPU per event: %f seconds"%np.mean(gpu_ml_per_event))
print("Average total time on GPU per event: %f seconds"%np.mean(gpu_total_per_event))

plt.figure(figsize=(10,7))
plt.hist(total_gpu,bins=50,label="Total Time")
plt.hist(gpu_ml,bins=50,label="Time for CNN only")
plt.xlabel("Total Time per File (s)",fontsize=20)
plt.title("Total Time Per File",fontsize=25)
plt.legend(fontsize=20)
plt.savefig("%sTimePerFile.png"%(save_folder))
plt.close()

plt.figure(figsize=(10,7))
plt.hist(gpu_total_per_event,bins=100,label="Total Time")
plt.hist(gpu_ml_per_event,bins=100,label="Time for CNN only")
plt.xlabel("Average Times per Event (s)",fontsize=20)
plt.title("Average Times Per Event",fontsize=25)
plt.legend(fontsize=20)
plt.savefig("%sTimePerEvent.png"%(save_folder))
plt.close()

plt.figure(figsize=(10,7))
plt.hist(energy_gpu/num_events,bins=50,label="Energy (first)")
plt.hist(class_gpu/num_events,bins=50,label="PID Class")
plt.hist(zenith_gpu/num_events,bins=50,label="Zenith")
plt.hist(vertex_gpu/num_events,bins=50,label="Vertex (x, y, z)")
plt.hist(muon_gpu/num_events,bins=50,label="Muon Class")
plt.xlabel("Average Time per CNN (s)",fontsize=20)
plt.title("Average Time per CNN",fontsize=25)
plt.legend(fontsize=20)
plt.savefig("%sTimePerType.png"%(save_folder))
plt.close()

if check_cpu:
    energy_cpu = np.genfromtxt("test1_cpu.txt")
    class_cpu = np.genfromtxt("test2_cpu.txt")
    zenith_cpu = np.genfromtxt("test3_cpu.txt")
    total_cpu =  np.genfromtxt("total_cpu.txt")
    num_events =  np.genfromtxt("events.txt")

    cpu_ml = energy_cpu + class_cpu + zenith_cpu + vertex_gpu
    cpu_ml_per_event = cpu_ml / num_events

    print("Average time for CPU per event: %f seconds"%np.mean(cpu_ml_per_event))
