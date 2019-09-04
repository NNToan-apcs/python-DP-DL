import matplotlib.pyplot as plt
import numpy as np
import re

def epoch_process(epoch_str):
    raw_epoch = epoch_str.split('/')
    epoch = raw_epoch[0].split(' ')[1]
    max_epoch = raw_epoch[1].split('\\')[0]
    
    return epoch, max_epoch

file_name = "mnist_dpsgd_100.txt"
file_name2 = "mnist_dpsgd_100_2.txt"
file_name3 = "mnist_dpsgd_100_3.txt"
file_name_sgd = "mnist_sgd_100.txt"
# file_name = "test1.txt"
#read data
f=open(file_name, "r")
if f.mode == 'r':
    contents = f.read()
    extracted_numbers = re.findall("\d+\.\d+", contents)
    meta_data = extracted_numbers[0:4]
    print(meta_data)
    extracted_numbers = extracted_numbers[4:]
    print(extracted_numbers)
    epsilons = extracted_numbers[0::2]
    epsilons = np.array(epsilons,dtype=float)
    model_acc = extracted_numbers[1::2]
    model_acc = np.array(model_acc,dtype=float)
    print(epsilons)
    print(model_acc)
f.close()

f=open(file_name2, "r")
if f.mode == 'r':
    contents = f.read()
    extracted_numbers2 = re.findall("\d+\.\d+", contents)
    meta_data2 = extracted_numbers2[0:4]
    print(meta_data2)
    extracted_numbers2 = extracted_numbers2[4:]
    print(extracted_numbers2)
    epsilons2 = extracted_numbers2[0::2]
    epsilons2 = np.array(epsilons2,dtype=float)
    model_acc2 = extracted_numbers2[1::2]
    model_acc2 = np.array(model_acc2,dtype=float)
    print(epsilons2)
    print(model_acc2)
f.close()

f=open(file_name3, "r")
if f.mode == 'r':
    contents = f.read()
    extracted_numbers3 = re.findall("\d+\.\d+", contents)
    meta_data3 = extracted_numbers3[0:4]
    print(meta_data3)
    extracted_numbers3 = extracted_numbers3[4:]
    print(extracted_numbers3)
    epsilons3 = extracted_numbers3[0::2]
    epsilons3 = np.array(epsilons3,dtype=float)
    model_acc3 = extracted_numbers3[1::2]
    model_acc3 = np.array(model_acc3,dtype=float)
    print(epsilons3)
    print(model_acc3)
f.close()

f=open(file_name_sgd, "r")
if f.mode == 'r':
    contents = f.read()
    extracted_numbers3 = re.findall("\d+\.\d+", contents)
    meta_data3 = extracted_numbers3[0:4]
    print(meta_data3)
    model_acc_sgd = extracted_numbers3[4:]
    model_acc_sgd = np.array(model_acc_sgd,dtype=float)
    print(model_acc_sgd)
f.close()
# plt.yticks(np.arange(0, 5, step=0.2))

# plt.axis([None, None, 0, 100])

# Plot 1
plot1 = plt.subplot(2, 1, 1)

plt.title('epsilon over epochs')
plt.xlabel('epochs')
plt.ylabel('epsilon')

xmajor_ticks = np.arange(0, 101, 20)
ymajor_ticks = np.arange(0, 11, 1)                                                                       
plot1.set_xticks(xmajor_ticks)                                                       
plot1.set_yticks(ymajor_ticks)                                                       
plot1.grid(which='both') 

line1, = plt.plot(range(1,len(epsilons)+1), epsilons, '.-')
line1.set_label('ηt=' + meta_data[0] + ', σ=' + meta_data[1] + ', C=' + meta_data[2]+ ', Batch Size='+ meta_data[3] + ', δ=1e-5')
line2, = plt.plot(range(1,len(epsilons2)+1), epsilons2, '.-')
line2.set_label('ηt=' + meta_data2[0] + ', σ=' + meta_data2[1] + ', C=' + meta_data2[2]+ ', Batch Size='+ meta_data2[3] + ', δ=1e-5')
line3, = plt.plot(range(1,len(epsilons3)+1), epsilons3, '.-')
line3.set_label('ηt=' + meta_data3[0] + ', σ=' + meta_data3[1] + ', C=' + meta_data3[2]+ ', Batch Size='+ meta_data3[3] + ', δ=1e-5')
plt.legend()
# Plot 2
plot2 = plt.subplot(2, 1, 2)

plt.title('accuracy over epochs')
plt.xlabel('epochs')
plt.ylabel('testing accuracy')

xmajor_ticks = np.arange(0, 101, 20)
ymajor_ticks = np.arange(0, 101, 10)                                                                                 
plot2.set_xticks(xmajor_ticks)                                                       
plot2.set_yticks(ymajor_ticks)                                                       
plot2.grid(which='both') 

plt.plot(range(1,len(model_acc)+1), model_acc, 'g.-')
line1, = plt.plot(range(1,len(model_acc)+1), model_acc, '.-')
line1.set_label('ηt=' + meta_data[0] + ', σ=' + meta_data[1] + ', C=' + meta_data[2]+ ', Batch Size='+ meta_data[3] + ', δ=1e-5')
line2, = plt.plot(range(1,len(model_acc2)+1), model_acc2, '.-')
line2.set_label('ηt=' + meta_data2[0] + ', σ=' + meta_data2[1] + ', C=' + meta_data2[2]+ ', Batch Size='+ meta_data2[3] + ', δ=1e-5')
line3, = plt.plot(range(1,len(model_acc3)+1), model_acc3, '.-')
line3.set_label('ηt=' + meta_data3[0] + ', σ=' + meta_data3[1] + ', C=' + meta_data3[2]+ ', Batch Size='+ meta_data3[3] + ', δ=1e-5')
line4, = plt.plot(range(1,len(model_acc_sgd)+1), model_acc_sgd, '.-')
line4.set_label('SGD')
plt.legend()
plt.show()

plot3 = plt.subplot(2, 1, 3)

