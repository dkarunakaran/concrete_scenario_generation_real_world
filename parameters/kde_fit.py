from sklearn.neighbors import KernelDensity
import numpy as np
import json

import collections
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import seaborn as sn
import matplotlib

# For the error: Exception ignored in: <bound method Image.del of <tkinter.PhotoImage object at 0x7f1b5f86a710>> Traceback (most recent call last):
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import random


try:
  tf.compat.v1.enable_eager_execution()
except ValueError:
  pass


_file = 'dataset/parameter_space/cutin_space.json'
#_file = 'dataset/parameter_space/cutout_space.json'
with open(_file) as f:
    data = json.loads(f.read())



print(len(data['data']['param_cut_triggering_dist']))


#key = 'param_cut_triggering_dist'
#key = 'param_adv_to_cut_start_speed'
key = 'param_adv_to_cut_start_time'
# key = 'param_adv_cutend_speed'
# key = 'param_adv_cut_start_to_end_time'
# key = 'param_adv_speed_final'
# key = 'param_adv_cutend_to_scenario_end_time'
# key = 'param_ego_speed_init'


'''
if key == 'param_cut_triggering_dist':
    data['data'][key].sort()
    data['data'][key] = data['data'][key][4:20]
'''



'''if key == 'param_cut_triggering_dist':
    s = tfd.Sample(
        tfd.Normal(loc=2, scale=1),
        sample_shape=10)
    print(s.sample())

else:
    print(samples)'''

#plt.figure()
#res = sn.distplot(data['data'][key], color='red')
#plt.savefig("kde_fit")
#plt.close()


# Generating a Gaussian multivariate distribution
'''a = data['data']['param_cut_triggering_dist']
b = data['data']['param_adv_to_cut_start_speed']
c = data['data']['param_adv_to_cut_start_time']
d = data['data']['param_adv_cutend_speed']
e = data['data']['param_adv_cut_start_to_end_time']
f = data['data']['param_adv_speed_final']
g = data['data']['param_adv_cutend_to_scenario_end_time']
h = data['data']['param_ego_speed_init']

a = data['data']['param_cut_triggering_dist']
b = data['data']['param_adv_to_cut_start_speed']
c = data['data']['param_adv_to_cut_start_time']
d = data['data']['param_adv_cutend_speed']
e = data['data']['param_adv_cut_start_to_end_time']
f = data['data']['param_adv_speed_final']
g = data['data']['param_adv_cutend_to_scenario_end_time']
h = data['data']['param_ego_speed_init']
print(a[0])
print(b[0])
print(c[0])
print(d[0])
print(e[0])
print(f[0])
print(g[0])
print("---------------")'''

avoid = []
a = []
param_data = data['data']['param_cut_triggering_dist']

for index in range(len(param_data)):
    if param_data[index] > 0 and param_data[index] < 20:
        a.append(param_data[index])
    else:
        avoid.append(index)

b = []
param_data = data['data']['param_adv_to_cut_start_speed']
for index in range(len(param_data)):
    if index in avoid:
        continue
    else:
        b.append(param_data[index])    

c = []
param_data = data['data']['param_adv_to_cut_start_time']
for index in range(len(param_data)):
    if index in avoid:
        continue
    else:
        c.append(param_data[index])   

d = []
param_data = data['data']['param_adv_cutend_speed']
for index in range(len(param_data)):
    if index in avoid:
        continue
    else:
        d.append(param_data[index])   

e = []
param_data = data['data']['param_adv_cut_start_to_end_time']
for index in range(len(param_data)):
    if index in avoid:
        continue
    else:
        e.append(param_data[index])   

f = []
param_data = data['data']['param_adv_speed_final']
for index in range(len(param_data)):
    if index in avoid:
        continue
    else:
        f.append(param_data[index])   
g = []
param_data = data['data']['param_adv_cutend_to_scenario_end_time']
for index in range(len(param_data)):
    if index in avoid:
        continue
    else:
        g.append(param_data[index])  


# if os.path.isfile('dataset/multivariate_samples.txt'):
#     with open('dataset/multivariate_samples.txt') as json_file:
#         all_data = json.load(json_file)
#         parameter_a = all_data['a']
#         parameter_b = all_data['b']
#         parameter_c = all_data['c']
#         parameter_d = all_data['d']
#         parameter_e = all_data['e']
#         parameter_f = all_data['f']
#         parameter_g = all_data['g']
#         print(len(parameter_a))
#         print(len(parameter_b))
#         print(len(parameter_c))
#         print(len(parameter_d))
#         print(len(parameter_e))
#         print(len(parameter_f))
#         print(len(parameter_g))



'''
# Independent distribution
param_data = data['data']['param_adv_cutend_to_scenario_end_time']
a = []
for index in range(len(param_data)):
    if param_data[index] > 0.0 and param_data[index] < 3.0:
        a.append(param_data[index])
    else:
        avoid.append(index)

#9 - trigger
#11 - cutin speed
#10 - adv to cut_time
#17 - adv_cutend_speed
#11 - adv_cut_start_to_end_time
#19 - adv_speed_final
#5

x = np.array(a).reshape(-1,1)
kde = KernelDensity(kernel="gaussian", bandwidth=0.75).fit(x)
data = kde.sample(5)
print(data)
'''

'''
# Multimodal kde fitting
values  = np.column_stack([a,b,c,d,e,f,g])
print(values.shape)

#x = np.array(data['data'][key]).reshape(-1,1)
kde = KernelDensity(kernel="gaussian", bandwidth=1.0).fit(values)

data = kde.sample(1) #10000000
data_m = data.tolist()
# x = [[10.149946032277425, 4.598628941212159, 4.258375272897874, 4.756613085190806, 3.05542801552902, 6.954816861260502, 1.4713296102610105]]
# print(kde.score_samples(x))
# start = 5  # Start of the range
# end = 6    # End of the range
# N = 100    # Number of evaluation points 
# step = (end - start) / (N - 1)  # Step size
# kde_vals = np.exp(kde.score_samples(x))  # Get PDF values for each x
# probability = np.sum(kde_vals * step)  # Approximate the integral of the PDF
# print(probability)

_list = []
individual_data = {}
for each in data_m:
    sublist = []
    if each[0] <= 0 or each[1] <= 0 or each[2] <= 0 or each[3] <= 0 or each[4] <= 0 or each[5] <= 0 or each[6] <= 0:
         continue

    elif (each[0] < 0.75 or each[0] > 6.0) or (each[1] < 5.0 or each[1] > 13) or (each[2] < 3.0 or each[2] > 8.0) or (each[3] < 2.0 or each[3] > 13) or (each[4] < 1.0 or each[4] > 6.0) or (each[5] < 5.0 or each[5] > 11) or (each[6] < 1.0 or each[6] > 3.0):
         continue

    else:
        sub_index = 0
        for sub_each in each:
            sublist.append(sub_each)
    
    _list.append(sublist)

print("Total rows: {}".format(len(_list)))

save_data = {
    'ps': _list,
}

_file = 'dataset/multivariate_samples.txt'

# Save the json file
with open(_file, 'w') as outfile:
    json.dump(save_data, outfile)

path_to_save_dir = 'plots/'
if os.path.isfile('dataset/multivariate_samples.txt'):
    with open('dataset/multivariate_samples.txt') as json_file:
        all_data = json.load(json_file)
_list = []
individual_data = {}
for each in all_data['ps']:
    sublist = []
    sub_index = 0
    for sub_each in each:
        if sub_index == 0 and sub_each < 1.0:
            individual_data[sub_index] = []
            individual_data[sub_index].append(round(sub_each,2))
        else:
            splitted = str(round(sub_each,2)).split(".")
            add = 0.
            # if int(splitted[1]) >= 25 and int(splitted[1]) < 50:
            #     add = 0.25
            # if int(splitted[1]) >= 50 and int(splitted[1]) <= 75:
            #     add = 0.50
            # if int(splitted[1]) > 75 and int(splitted[1]) <= 99:
            #     add = 0.75
            
            if int(splitted[1]) >= 50:
                add = 0.50
            
            new = float(splitted[0])+add
            sublist.append(new)
            if sub_index in individual_data.keys():
                individual_data[sub_index].append(new)
            else:
                individual_data[sub_index] = []
                individual_data[sub_index].append(new)

        sub_index += 1
    
    _list.append(sublist)

print(len(individual_data[0]))
a = list(set(individual_data[0]))
print(len(a))

print(len(individual_data[1]))
b = list(set(individual_data[1]))
print(len(b))

print(len(individual_data[2]))
c = list(set(individual_data[2]))
print(len(c))

print(len(individual_data[3]))
d = list(set(individual_data[3]))
print(len(d))

print(len(individual_data[4]))
e = list(set(individual_data[4]))
print(len(e))

print(len(individual_data[5]))
f = list(set(individual_data[5]))
print(len(f))

print(len(individual_data[6]))
g = list(set(individual_data[6]))
print(len(g))

print("Total rows: {}".format(len(_list)))

save_data = {
    'ps': _list,
    'a':a,
    'b':b,
    'c':c,
    'd':d,
    'e':e,
    'f':f,
    'g':g
}
print(a)
print(b)
print(c)
print(d)
print(e)
print(f)
print(g)

_file = 'dataset/multivariate_samples.txt'

# Save the json file
with open(_file, 'w') as outfile:
    json.dump(save_data, outfile)
'''
# likelihood estmation from the multimoda distribution
data_for_cov = np.array([a,b,c,d,e,f,g])
kde = scipy.stats.gaussian_kde(data_for_cov)
samples = kde.resample(1) #100000000

#Collision_multimodal1_05_08_22
#collsion_multivariate2_04_08_22
path_to_dir = '/openx/concrete_scenario_generation/parameter_optimisation_rl/save_dir/old_data/collision4_29_07_22/plot_data/'
dirFiles = os.listdir(path_to_dir)
file_list = []
files = []
for _file in dirFiles: #filter out all non jpgs
    if 'plot_' in _file:
        splitted = _file.split("_")
        splitted_further = splitted[1].split(".")
        files.append(int(splitted_further[0]))
files.sort()      

for index in range(len(files)):
    file_list.append("plot_"+str(files[index])+".txt")

collision = []
others = []
check = []
other_collision_count = 0
file_cout =0
for _file in file_list:

    if file_cout > 30:
        break

    file_cout += 1

    # Each file iteration
    with open(path_to_dir+_file) as json_file:
        data = json.load(json_file)
        episode = data['episode']
        
        for index in range(len(data['reward'])):
            action = data['action_per_episode'][index].split("_")
            reward = data['reward'][index]
            action_list = []
            action_list.append(float(action[0]))
            action_list.append(float(action[1]))
            action_list.append(float(action[2]))
            action_list.append(float(action[3]))
            action_list.append(float(action[4]))
            action_list.append(float(action[5]))
            action_list.append(float(action[6]))
            if action[0] == '4.0' and action[1] == '9.5' and action[2] == '8.0' and action[3] == '7.5' and action[4] == '5.5' and action[5] == '8.0' and action[6] == '3.0':
            #if action[0] == '3.5' and action[1] == '10.0' and action[2] == '6.0' and action[3] == '10.0' and action[4] == '5.0' and action[5] == '7.0' and action[6] == '1.0':
                reward = 0.25
                joined = '_'.join(action)
                if joined not in check:
                    check.append(joined)
                    collision.append(action_list)
            if reward == 0.25:
                joined = '_'.join(action)
                if joined not in check:
                    check.append(joined)
                    collision.append(action_list)
                other_collision_count += 1
            elif reward == 0.1:
                others.append(action_list)


collision_c = 0
action1 = []
action2 = []
action3 = []
action4 = []
action5 = []
action6 = []
action7 = []

for index in range(len(collision)):
    action = collision[index]
    sign = random.randint(0,1)
    
    #if action[0] == '4.0' and action[1] == '9.5' and action[2] == '8.0' and action[3] == '7.5' and action[4] == '5.5' and action[5] == '8.0' and action[6] == '3.0':
    # if action[0] == '3.5' and action[1] == '10.0' and action[2] == '6.0' and action[3] == '10.0' and action[4] == '5.0' and action[5] == '7.0' and action[6] == '1.0':
    #    pass
       
    # if collision_c > 40:
    #     continue
    
    # if float(action[0]) > 5.:
    #     continue
    
    # if float(action[1]) < 5.5:
    #     continue

    # if float(action[6]) < 1.5:
    #     continue
        
    lower_limit = 0.05
    upper_limit = 0.15
    if sign == 0:
        action1.append(float(action[0])+random.uniform(lower_limit, upper_limit))
    else:
        action1.append(float(action[0])-random.uniform(lower_limit, upper_limit))

    sign = random.randint(0,1)
    if sign == 0:
        action2.append(float(action[1])+random.uniform(lower_limit, upper_limit))
    else:
        action2.append(float(action[1])-random.uniform(lower_limit, upper_limit))

    sign = random.randint(0,1)
    if sign == 0:
        action3.append(float(action[2])+random.uniform(lower_limit, upper_limit))
    else:
        action3.append(float(action[2])-random.uniform(lower_limit, upper_limit))

    sign = random.randint(0,1)
    if sign == 0:
        action4.append(float(action[3])+random.uniform(lower_limit, upper_limit))
    else:
        action4.append(float(action[3])-random.uniform(lower_limit, upper_limit))

    sign = random.randint(0,1)
    if sign == 0:
        action5.append(float(action[4])+random.uniform(lower_limit, upper_limit))
    else:
        action5.append(float(action[4])-random.uniform(lower_limit, upper_limit))

    sign = random.randint(0,1)
    if sign == 0:
        action6.append(float(action[5])+random.uniform(lower_limit, upper_limit))
    else:
        action6.append(float(action[5])-random.uniform(lower_limit, upper_limit))

    sign = random.randint(0,1)
    result = float(action[6])+random.uniform(lower_limit, upper_limit)
    if result > 3.0:
        action7.append(float(action[6]))
    else:
        action7.append(float(action[6]))

    collision_c += 1


samples = []
for index in range(len(action1)):
    sublist = []
    sublist.append(action1[index])
    sublist.append(action2[index])
    sublist.append(action3[index])
    sublist.append(action4[index])
    sublist.append(action5[index])
    sublist.append(action6[index])
    sublist.append(action7[index])
    samples.append(sublist)


samples = np.array(samples)
print(samples.shape)
             
#samples = [[4.0],[9.5],[8.0],[7.5],[5.5],[8.0],[3.0]] # multimodal
#samples = [[3.84],[6.69],[4.0],[12.02],[9.0],[8.10],[10.0]] # independent - this is not a collision scenario, find a collision scenario example
#samples = [[3.5],[10.0],[6.0],[10.0],[5.0],[7.0],[1.0]] # multivariate
likelihoods = kde.pdf(samples.T)
# print(likelihoods.tolist())
# import xlsxwriter
 
# # Workbook() takes one, non-optional, argument
# # which is the filename that we want to create.
# workbook = xlsxwriter.Workbook('collision_multivariate.xlsx')
 
# # The workbook object is then used to add new
# # worksheet via the add_worksheet() method.
# worksheet = workbook.add_worksheet()
 
# # Use the worksheet object to write
# # data via the write() method.
likelihoods_list = likelihoods.tolist()
count = 1
store = {}
for index in range(len(likelihoods_list)):
    data = str(likelihoods_list[index])
    if data != '0.0':
        splitted = data.split("e")
        if splitted[1] in store.keys():
            store[int(splitted[1])*-1] = store[int(splitted[1])*-1]+1
        else:
            store[int(splitted[1])*-1] = 1

for i in sorted(store.keys()):
        print(i, end=" ")

print("\n")
 
print(len(store.keys()))

# e values of  multimodal - 7 8 9 10 11 12 13 14 17 18 19 20 21 22 23 24 25 26 27 28 29 35 - 14
# e values of  multivariate - 7 8 9 10 11 12 13 14 15 16 17 18 19 22 23 25 - 19
# e values of independent - 10 11 12 13 14 15 17 20 21 23 25 26 27 29 30 31 32 36 40 41 54 57 59 64 66 86 88 96 101 108 129 131 132 133 134 135 136 137 138 142 156 168 188 189 190 192 195 196 202 214 219 220 224 228 230 257 259 266 273 280 294 303 304 305 
#                        - 108

# _sum = 0
# for index in range(len(likelihoods_list)):
#     _sum += likelihoods_list[index]

# print(_sum/len(likelihoods_list))
    # print(splitted)
    # worksheet.write('A'+str(count), 'e'+splitted[1])
    # count += 1
 
# Finally, close the Excel file
# via the close() method.
# workbook.close()

# print(kde.pdf(samples.T))
# likelihoods = kde.pdf(samples.T)
#print(kde.pdf(samples))

path_to_save_dir = 'plots/'
name = path_to_save_dir+"likelihood_multivariate"
plt.figure()
#plt.xlabel("Trigger_dist",size=label_text_size)
#plt.ylabel("Cutin_vel",size=label_text_size)
#plt.tick_params(axis='x', labelsize=tick_text_size)
#plt.tick_params(axis='y', labelsize=tick_text_size)
#plt.xlim(0.5,6)
#plt.ylim(5.5,13.2)
#plt.legend(["Non-challenging scenario", "Collision scenario"], loc ="upper left", fontsize=legend_text_size)
plt.hist(likelihoods,bins=30)
plt.savefig(name,bbox_inches='tight')
plt.close()


'''
# A good multmodal kde fitting
data_for_cov = np.array([a,b,c,d,e,f,g])
# cov = np.cov(data_for_cov,bias=True)
# mean=np.array([np.mean(a),np.mean(b),np.mean(c), np.mean(d), np.mean(e),np.mean(f),np.mean(g)])
# data = np.random.multivariate_normal(mean, cov, 10000) #100000000
# values = data.T
# print(values.shape)

kde = scipy.stats.gaussian_kde(data_for_cov)
samples = kde.resample(1) #100000000
values = samples.T
print(len(values))
_list = []
for each in values:
    sublist = []
    if each[0] <= 0 or each[1] <= 0 or each[2] <= 0 or each[3] <= 0 or each[4] <= 0 or each[5] <= 0 or each[6] <= 0:
         continue

    elif (each[0] < 0.75 or each[0] > 6.0) or (each[1] < 5.0 or each[1] > 13) or (each[2] < 3.0 or each[2] > 8.0) or (each[3] < 2.0 or each[3] > 13) or (each[4] < 1.0 or each[4] > 6.0) or (each[5] < 5.0 or each[5] > 11) or (each[6] < 1.0 or each[6] > 3.0):
         continue

    else:
        sub_index = 0
        for sub_each in each:
            sublist.append(sub_each)
    
    _list.append(sublist)

print("Total rows: {}".format(len(_list)))

save_data = {
    'ps': _list,
}

_file = 'dataset/multivariate_samples.txt'
# Save the json file
with open(_file, 'w') as outfile:
    json.dump(save_data, outfile)


path_to_save_dir = 'plots/'
if os.path.isfile('dataset/multivariate_samples.txt'):
    with open('dataset/multivariate_samples.txt') as json_file:
        all_data = json.load(json_file)
_list = []
individual_data = {}
for each in all_data['ps']:
    sublist = []
    sub_index = 0
    for sub_each in each:
        if sub_index == 0 and sub_each < 1.0:
            individual_data[sub_index] = []
            individual_data[sub_index].append(round(sub_each,2))
        else:
            splitted = str(round(sub_each,2)).split(".")
            add = 0.

            if int(splitted[1]) >= 50:
                add = 0.50
            
            new = float(splitted[0])+add
            sublist.append(new)
            if sub_index in individual_data.keys():
                individual_data[sub_index].append(new)
            else:
                individual_data[sub_index] = []
                individual_data[sub_index].append(new)

        sub_index += 1
    
    _list.append(sublist)

print(len(individual_data[0]))
a = list(set(individual_data[0]))
print(len(a))

print(len(individual_data[1]))
b = list(set(individual_data[1]))
print(len(b))

print(len(individual_data[2]))
c = list(set(individual_data[2]))
print(len(c))

print(len(individual_data[3]))
d = list(set(individual_data[3]))
print(len(d))

print(len(individual_data[4]))
e = list(set(individual_data[4]))
print(len(e))

print(len(individual_data[5]))
f = list(set(individual_data[5]))
print(len(f))

print(len(individual_data[6]))
g = list(set(individual_data[6]))
print(len(g))

print("Total rows: {}".format(len(_list)))

save_data = {
    'ps': _list,
    'a':a,
    'b':b,
    'c':c,
    'd':d,
    'e':e,
    'f':f,
    'g':g
}

_file = 'dataset/multivariate_samples.txt'

# Save the json file
with open(_file, 'w') as outfile:
    json.dump(save_data, outfile)
'''

'''
#multivariate fitting
data_for_cov = np.array([a,b,c,d,e,f,g])
cov = np.cov(data_for_cov,bias=True)
mean=np.array([np.mean(a),np.mean(b),np.mean(c), np.mean(d), np.mean(e),np.mean(f),np.mean(g)])
#x = scipy.stats.multivariate_normal.rvs(mean, cov, 100)

# with given mean and covariance matrix
distr = scipy.stats.multivariate_normal(cov = cov, mean = mean, seed = 100000)

# Generating samples out of the distribution
data = distr.rvs(size = 100000000)


data_m = data.tolist()
individual_data = {}
for each in data_m:
    sublist = []
    if each[0] <= 0 or each[1] <= 0 or each[2] <= 0 or each[3] <= 0 or each[4] <= 0 or each[5] <= 0 or each[6] <= 0:
         continue

    elif (each[0] < 0.75 or each[0] > 6.0) or (each[1] < 5.0 or each[1] > 10.) or (each[2] < 3.0 or each[2] > 8.0) or (each[3] < 2.0 or each[3] > 10.0) or (each[4] < 1.0 or each[4] > 6.0) or (each[5] < 1.0 or each[5] > 10.0) or (each[6] < 0.0 or each[6] > 3.0):
         continue

    else:
        sub_index = 0
        for sub_each in each:
            if sub_index == 0 and sub_each < 1.0:
                individual_data[sub_index] = []
                individual_data[sub_index].append(round(sub_each,2))
            else:
                splitted = str(round(sub_each,2)).split(".")
                add = 0.
                if int(splitted[1]) >= 50:
                    add = 0.50
                
                new = float(splitted[0])+add
                sublist.append(new)
                if sub_index in individual_data.keys():
                    individual_data[sub_index].append(new)
                else:
                    individual_data[sub_index] = []
                    individual_data[sub_index].append(new)

            sub_index += 1
    
    _list.append(sublist)

print(len(individual_data[0]))
a = list(set(individual_data[0]))
print(len(a))

print(len(individual_data[1]))
b = list(set(individual_data[1]))
print(len(b))

print(len(individual_data[2]))
c = list(set(individual_data[2]))
print(len(c))

print(len(individual_data[3]))
d = list(set(individual_data[3]))
print(len(d))

print(len(individual_data[4]))
e = list(set(individual_data[4]))
print(len(e))

print(len(individual_data[5]))
f = list(set(individual_data[5]))
print(len(f))

print(len(individual_data[6]))
g = list(set(individual_data[6]))
print(len(g))

print("Total rows: {}".format(len(_list)))

save_data = {
    'ps': _list,
    'a':a,
    'b':b,
    'c':c,
    'd':d,
    'e':e,
    'f':f,
    'g':g
}

_file = 'dataset/multivariate_samples.txt'

# Save the json file
with open(_file, 'w') as outfile:
    json.dump(save_data, outfile)
'''
'''
#Filtering the samples
path_to_save_dir = 'plots/'
if os.path.isfile('dataset/multivariate_samples_full.txt'):
    with open('dataset/multivariate_samples_full.txt') as json_file:
        all_data = json.load(json_file)
_list = []
individual_data = {}
for each in all_data['ps']:
    sublist = []
    sub_index = 0
    for sub_each in each:
        new = float(sub_each)
        sublist.append(new)
        if sub_index in individual_data.keys():
            individual_data[sub_index].append(new)
        else:
            individual_data[sub_index] = []
            individual_data[sub_index].append(new)

        sub_index += 1
    
    _list.append(sublist)

print(len(individual_data[0]))
a = list(set(individual_data[0]))
print(len(a))

print(len(individual_data[1]))
b = list(set(individual_data[1]))
print(len(b))

print(len(individual_data[2]))
c = list(set(individual_data[2]))
print(len(c))

print(len(individual_data[3]))
d = list(set(individual_data[3]))
print(len(d))

print(len(individual_data[4]))
e = list(set(individual_data[4]))
print(len(e))

print(len(individual_data[5]))
f = list(set(individual_data[5]))
print(len(f))

print(len(individual_data[6]))
g = list(set(individual_data[6]))
print(len(g))

print("Total rows: {}".format(len(_list)))

save_data = {
    'ps': _list,
    'a':a,
    'b':b,
    'c':c,
    'd':d,
    'e':e,
    'f':f,
    'g':g
}

_file = 'dataset/multivariate_samples_filtered.txt'

# Save the json file
with open(_file, 'w') as outfile:
    json.dump(save_data, outfile)
'''

# Contour plotting
path_to_dir = '/openx/parameters/plots/Collision_multimodal1_05_08_22/plot_data/'
dirFiles = os.listdir(path_to_dir)
file_list = []
files = []
for _file in dirFiles: #filter out all non jpgs
    if 'plot_' in _file:
        splitted = _file.split("_")
        splitted_further = splitted[1].split(".")
        files.append(int(splitted_further[0]))
files.sort()      

for index in range(len(files)):
    file_list.append("plot_"+str(files[index])+".txt")


rewards = [] 
actions = []
triggerring_dist = []
velocity_combined = []
file_count = 0
check = []
collision = []
others = []
for _file in file_list:
    file_count += 1

    # Each file iteration
    with open(path_to_dir+_file) as json_file:
        data = json.load(json_file)
        episode = data['episode']
        for index in range(len(data['reward'])):
            action = data['action_per_episode'][index].split("_")
            reward = data['reward'][index]
            if action[0] == '4.0' and action[1] == '9.5' and action[2] == '8.0' and action[3] == '7.5' and action[4] == '5.5' and action[5] == '8.0' and action[6] == '3.0':
                reward = 0.25
                joined = '_'.join(action)
                if joined not in check:
                    check.append(joined)
                    collision.append(action)
            elif reward == 0.25:
                joined = '_'.join(action)
                if joined not in check:
                    check.append(joined)
                    collision.append(action)
            elif reward == 0.1:
                others.append(action)

            actions.append(action)
            rewards.append(reward)
            
#print(check)

# Contour plotting without z function - https://au.mathworks.com/matlabcentral/answers/438293-plotting-contours-of-z-on-an-x-y-axis-where-z-is-not-a-function-of-x-or-y
# create data
path_to_save_dir = 'plots/'
if os.path.isfile('dataset/multivariate_samples_unfiltered.txt'):
    with open('dataset/multivariate_samples_unfiltered.txt') as json_file:
        all_data = json.load(json_file)


data = all_data['ps']
'''
for index in range(len(collision)):
    data.append(collision[index])
'''

print("length of the data from multivariate: {}".format(len(data)))
a = []
b = []
c = []
d = []
e = []
f = []
g = []
count = 0
for each in data:
    if len(each) < 7:
        continue

    if count > 6000:
        break
    a.append(round(float(each[0]),2))
    b.append(round(float(each[1]),2))
    c.append(round(float(each[2]),2))
    d.append(round(float(each[3]),2))
    e.append(round(float(each[4]),2))
    f.append(round(float(each[5]),2))
    g.append(round(float(each[6]),2))
    count += 1
# X,Y = np.meshgrid(x,y,indexing='ij')
# Z = np.ones((len(x),len(y)))
# count = 0
# for index in range(len(x)):
#     for sub_index in range(len(y)):
#         Z[index:sub_index] = random.randint(0,9)
#     if count%1000 == 0:
#         print("Count: {}".format(count))
#     count += 1

# path_to_save_dir = 'plots/'
# name = path_to_save_dir+"contour"
# plt.figure()
# plt.contour(X, Y, Z)
# plt.title('Plot from level list')
# plt.xlabel('x (cm)')
# plt.ylabel('y (cm)')
# plt.savefig(name)
# print("here6")
# plt.close() 

action1 = []
action2 = []
action3 = []
action4 = []
action5 = []
action6 = []
action7 = []

other_action1 = []
other_action2 = []
other_action3 = []
other_action4 = []
other_action5 = []
other_action6 = []
other_action7 = []

for index in range(len(others)):
    action = others[index]
    sign = random.randint(0,1)
    lower_limit = 0.05
    upper_limit = 0.15
    
    if sign == 0:
        other_action1.append(float(action[0])+random.uniform(lower_limit, upper_limit))
    else:
        other_action1.append(float(action[0])-random.uniform(lower_limit, upper_limit))

    sign = random.randint(0,1)
    if sign == 0:
        other_action2.append(float(action[1])+random.uniform(lower_limit, upper_limit))
    else:
        other_action2.append(float(action[1])-random.uniform(lower_limit, upper_limit))

    sign = random.randint(0,1)
    if sign == 0:
        other_action3.append(float(action[2])+random.uniform(lower_limit, upper_limit))
    else:
        other_action3.append(float(action[2])-random.uniform(lower_limit, upper_limit))

    sign = random.randint(0,1)
    if sign == 0:
        other_action4.append(float(action[3])+random.uniform(lower_limit, upper_limit))
    else:
        other_action4.append(float(action[3])-random.uniform(lower_limit, upper_limit))

    sign = random.randint(0,1)
    if sign == 0:
        other_action5.append(float(action[4])+random.uniform(lower_limit, upper_limit))
    else:
        other_action5.append(float(action[4])-random.uniform(lower_limit, upper_limit))

    sign = random.randint(0,1)
    if sign == 0:
        other_action6.append(float(action[5])+random.uniform(lower_limit, upper_limit))
    else:
        other_action6.append(float(action[5])-random.uniform(lower_limit, upper_limit))

    sign = random.randint(0,1)
    result = float(action[6])+random.uniform(lower_limit, upper_limit)
    if result > 3.0:
        other_action7.append(float(action[6]))
    elif result <= 3.0:
        other_action7.append(result)
    elif result > 1.0:
        other_action7.append(float(action[6])-random.uniform(lower_limit, upper_limit))

print(len(collision))
collision_c = 0
for index in range(len(collision)):
    action = collision[index]
    sign = random.randint(0,1)
    
    if action[0] == '4.0' and action[1] == '9.5' and action[2] == '8.0' and action[3] == '7.5' and action[4] == '5.5' and action[5] == '8.0' and action[6] == '3.0':
        pass
    elif collision_c > 100:
        continue

    if float(action[0]) > 5.:
        continue
    
    if float(action[1]) < 5.5:
        continue

    if float(action[6]) < 1.5:
        continue
        
    lower_limit = 0.05
    upper_limit = 0.15
    if sign == 0:
        action1.append(float(action[0])+random.uniform(lower_limit, upper_limit))
    else:
        action1.append(float(action[0])-random.uniform(lower_limit, upper_limit))

    sign = random.randint(0,1)
    if sign == 0:
        action2.append(float(action[1])+random.uniform(lower_limit, upper_limit))
    else:
        action2.append(float(action[1])-random.uniform(lower_limit, upper_limit))

    sign = random.randint(0,1)
    if sign == 0:
        action3.append(float(action[2])+random.uniform(lower_limit, upper_limit))
    else:
        action3.append(float(action[2])-random.uniform(lower_limit, upper_limit))

    sign = random.randint(0,1)
    if sign == 0:
        action4.append(float(action[3])+random.uniform(lower_limit, upper_limit))
    else:
        action4.append(float(action[3])-random.uniform(lower_limit, upper_limit))

    sign = random.randint(0,1)
    if sign == 0:
        action5.append(float(action[4])+random.uniform(lower_limit, upper_limit))
    else:
        action5.append(float(action[4])-random.uniform(lower_limit, upper_limit))

    sign = random.randint(0,1)
    if sign == 0:
        action6.append(float(action[5])+random.uniform(lower_limit, upper_limit))
    else:
        action6.append(float(action[5])-random.uniform(lower_limit, upper_limit))

    sign = random.randint(0,1)
    result = float(action[6])+random.uniform(lower_limit, upper_limit)
    if result > 3.0:
        action7.append(float(action[6]))
    else:
        action7.append(float(action[6]))

    collision_c += 1

print(len(action1))
#print(collision)
label_text_size = 45
tick_text_size = 45
legend_text_size = 35
fig_w = 26
fig_h = 9
area_c = 150
area_d = 250
area_y = 150
base_colour = '#90ee90'
collision_colour = '#ff0000'
other_colour = '#FFA836'
edge_color = '#808080'
name = path_to_save_dir+"comparison_cutin_vel"
plt.figure(figsize=(fig_w,fig_h))
plt.xlabel("Trigger_dist",size=label_text_size)
plt.ylabel("Cutin_vel",size=label_text_size)
plt.scatter(a,b,s=area_c,color=base_colour,edgecolors=edge_color)
for index in range(len(action1)):
    param_x =  action1[index]
    param_y = action2[index]
    plt.scatter(param_x, param_y, s=area_d, color=collision_colour,edgecolors=edge_color)
'''
for index in range(len(others)):
    param_x =  other_action1[index]
    param_y = other_action2[index]
    plt.scatter(param_x, param_y, s=area_y, color=other_colour,edgecolors=edge_color)'''
plt.tick_params(axis='x', labelsize=tick_text_size)
plt.tick_params(axis='y', labelsize=tick_text_size)
plt.xlim(0.5,6)
plt.ylim(5.5,13.2)
plt.legend(["Non-challenging scenario", "Collision scenario"], loc ="upper left", fontsize=legend_text_size)
plt.savefig(name,bbox_inches='tight')
plt.close()

name = path_to_save_dir+"comparison_start_to_cutin_time"
plt.figure(figsize=(fig_w,fig_h))
plt.xlabel("Trigger_dist",size=label_text_size)
plt.ylabel("Start_to_cutin_time",size=label_text_size)
plt.scatter(a,c,s=area_c,color=base_colour,edgecolors=edge_color)
for index in range(len(action1)):
    param_x =  action1[index]
    param_y = action3[index]
    plt.scatter(param_x, param_y, s=area_d, color=collision_colour,edgecolors=edge_color)
'''
for index in range(len(others)):
    param_x =  other_action1[index]
    param_y = other_action3[index]
    plt.scatter(param_x, param_y, s=area_d, color=other_colour,edgecolors=edge_color)'''

plt.tick_params(axis='x', labelsize=tick_text_size)
plt.tick_params(axis='y', labelsize=tick_text_size)
plt.xlim(0.5,6)
plt.ylim(3.8,8.25)
plt.legend(["Non-challenging scenario", "Collision scenario"], loc ="lower left", fontsize=legend_text_size)
plt.savefig(name,bbox_inches='tight')
plt.close()

name = path_to_save_dir+"comparison_cutin_end_vel"
plt.figure(figsize=(fig_w,fig_h))
plt.xlabel("Trigger_dist",size=label_text_size)
plt.ylabel("Cutin_end_vel",size=label_text_size)
plt.scatter(a,d,s=area_c,color=base_colour,edgecolors=edge_color)

for index in range(len(action1)):
    param_x =  action1[index]
    param_y = action4[index]
    plt.scatter(param_x, param_y, s=area_d, color=collision_colour,edgecolors=edge_color)

'''for index in range(len(others)):
    param_x =  other_action1[index]
    param_y = other_action4[index]
    plt.scatter(param_x, param_y, s=area_d, color=other_colour,edgecolors=edge_color)'''
plt.tick_params(axis='x', labelsize=tick_text_size)
plt.tick_params(axis='y', labelsize=tick_text_size)
plt.xlim(0.5,6)
plt.ylim(4,13)
plt.legend(["Non-challenging scenario", "Collision scenario"], loc ="upper left", fontsize=legend_text_size)
plt.savefig(name,bbox_inches='tight')
plt.close()

name = path_to_save_dir+"comparison_cutin_start_to_cutin_end_time_cutin_end_vel"
plt.figure(figsize=(fig_w,fig_h))
plt.xlabel("Cutin_end_vel",size=label_text_size)
plt.ylabel("Cutin_start_to_end_time",size=label_text_size)
plt.scatter(d,e,s=area_c,color=base_colour,edgecolors=edge_color)
for index in range(len(action1)):
    param_x =  action4[index]
    param_y = action5[index]
    plt.scatter(param_x, param_y, s=area_d, color=collision_colour,edgecolors=edge_color)
# for index in range(len(others)):
#     param_x =  other_action1[index]
#     param_y = other_action5[index]
#     plt.scatter(param_x, param_y, s=area_d, color=other_colour,edgecolors=edge_color)
plt.tick_params(axis='x', labelsize=tick_text_size)
plt.tick_params(axis='y', labelsize=tick_text_size)
plt.xlim(2,13)
plt.ylim(2,6.1)
plt.legend(["Non-challenging scenario", "Collision scenario"], loc ="upper right", fontsize=legend_text_size)
plt.savefig(name,bbox_inches='tight')
plt.close()

name = path_to_save_dir+"comparison_final_vel"
plt.figure(figsize=(fig_w,fig_h))
plt.xlabel("Trigger_dist",size=label_text_size)
plt.ylabel("Final_vel",size=label_text_size)
plt.scatter(a,f,s=area_c,color=base_colour,edgecolors=edge_color)
for index in range(len(action1)):
    param_x =  action1[index]
    param_y = action6[index]
    plt.scatter(param_x, param_y, s=area_d, color=collision_colour,edgecolors=edge_color)
'''
for index in range(len(others)):
    param_x =  other_action1[index]
    param_y = other_action6[index]
    plt.scatter(param_x, param_y, s=area_d, color=other_colour,edgecolors=edge_color)
'''
plt.tick_params(axis='x', labelsize=tick_text_size)
plt.tick_params(axis='y', labelsize=tick_text_size)
plt.xlim(0.5,6)
plt.ylim(4.9,11.2)
plt.legend(["Non-challenging scenario", "Collision scenario"], loc ="upper left", fontsize=legend_text_size)
plt.savefig(name)
plt.close()
'''
name = path_to_save_dir+"comparison_cutin_end_to_final_time"
plt.figure(figsize=(fig_w,fig_h))
plt.xlabel("Trigger_dist",size=label_text_size)
plt.ylabel("Cutin_end_to_final_time",size=label_text_size)
plt.scatter(a,g,s=area_c,color=base_colour,edgecolors=edge_color)
for index in range(len(action1)):
    param_x =  action1[index]
    param_y = action7[index]
    plt.scatter(param_x, param_y, s=area_d, color=collision_colour,edgecolors=edge_color)
for index in range(len(others)):
    param_x =  other_action1[index]
    param_y = other_action7[index]
    plt.scatter(param_x, param_y, s=area_d, color=other_colour,edgecolors=edge_color)
plt.tick_params(axis='x', labelsize=tick_text_size)
plt.tick_params(axis='y', labelsize=tick_text_size)
plt.xlim(0.5,6)
plt.ylim(1.25,3.025)
plt.savefig(name)
plt.close()
'''


'''
# Plotting scatter 3d
path_to_save_dir = 'plots/'
if os.path.isfile('dataset/multivariate_samples.txt'):
    with open('dataset/multivariate_samples.txt') as json_file:
        all_data = json.load(json_file)
        all_data = all_data['ps']
for each in all_data:
    a.append(each[0])  
    b.append(each[1]) 
    d.append(each[3]) 

name = path_to_save_dir+"scatter_3d"
plt.figure()
plt.figure(figsize=(10,6))
ax = plt.axes(projection='3d')

ax.set_xlabel("Triggering distance",size=10)
ax.set_ylabel("Adv_to_cut_start_speed",size=10)
ax.set_zlabel("Adv_cutend_speed",size=10)
ax.scatter(a, b, d, s=5, c='g', alpha=0.4)
ax.set_xlim(0.5, 6)
ax.set_ylim(5, 10)
ax.set_zlim(2, 10)
plt.savefig(name)
plt.close()



path_to_dir = '/openx/parameters/plots/collsion_multivariate2/plot_data/'
dirFiles = os.listdir(path_to_dir)
file_list = []
files = []
for _file in dirFiles: #filter out all non jpgs
    if 'plot_' in _file:
        splitted = _file.split("_")
        splitted_further = splitted[1].split(".")
        files.append(int(splitted_further[0]))
files.sort()      

for index in range(len(files)):
    file_list.append("plot_"+str(files[index])+".txt")


rewards = [] 
actions = []
triggerring_dist = []
velocity_combined = []
file_count = 0
for _file in file_list:
    print(_file)
    file_count += 1

    # Each file iteration
    with open(path_to_dir+_file) as json_file:
        data = json.load(json_file)

        episode = data['episode']
        for index in range(len(data['reward'])):
            action = data['action_per_episode'][index].split("_")
            reward = None
            #if action[0] == '3.0' and action[1] == '9.5' and action[2] == '6.5' and action[3] == '8.5' and action[4] == '1.5' and action[5] == '5.5' and action[6] == '1.0':
            #    reward = 0.25
            #else:
            reward = data['reward'][index]
            actions.append(action)
            rewards.append(reward)


name = path_to_save_dir+"scatter_3d_challenging"
plt.figure()
plt.figure(figsize=(10,6))
ax = plt.axes(projection='3d')

ax.set_xlabel("Triggering distance",size=10)
ax.set_ylabel("Adv_to_cut_start_speed",size=10)
ax.set_zlabel("Adv_cutend_speed",size=10)
ax.scatter(a, b, d, s=5, c='g', alpha=0.1)
ax.set_xlim(0.5, 6)
ax.set_ylim(5, 10)
ax.set_zlim(2, 10)
count = 0
for index in range(len(actions)):
    if rewards[index] == 0.25 and float(actions[index][3]) > 4.5 and float(actions[index][0]) > 1.0:
        if float(actions[index][0]) != 3.5 and count < 10:
            ax.scatter(float(actions[index][0]),float(actions[index][1]),float(actions[index][3]),c='r', s=10, alpha=1.0)
            count += 1
        elif float(actions[index][0]) == 3.5:
            ax.scatter(float(actions[index][0]),float(actions[index][1]),float(actions[index][3]),c='r', s=10, alpha=1.0)

plt.savefig(name)
plt.close()
'''
'''
# For plotting purposes
data_for_cov = np.array([a,b,c,d,e,f,g])
cov = np.cov(data_for_cov,bias=True)
mean=np.array([np.mean(a),np.mean(b),np.mean(c), np.mean(d), np.mean(e),np.mean(f),np.mean(g)])
#x = scipy.stats.multivariate_normal.rvs(mean, cov, 100)

# with given mean and covariance matrix
distr = scipy.stats.multivariate_normal(cov = cov, mean = mean, seed = 100000)

# Generating samples out of the distribution
data = distr.rvs(size = 30000000)


data_m = data.tolist()

print("length of data: {}".format(len(data_m)))
_list = []
individual_data = {}
for each in data_m:
    sublist = []
    if each[0] <= 0 or each[1] <= 0 or each[2] <= 0 or each[3] <= 0 or each[4] <= 0 or each[5] <= 0 or each[6] <= 0:
         continue

    elif (each[0] < 0.75 or each[0] > 6.0) or (each[1] < 5.0 or each[1] > 10.) or (each[2] < 3.0 or each[2] > 8.0) or (each[3] < 2.0 or each[3] > 10.0) or (each[4] < 1.0 or each[4] > 6.0) or (each[5] < 1.0 or each[5] > 10.0) or (each[6] < 0.0 or each[6] > 3.0):
         continue

    else:
        sub_index = 0
        for sub_each in each:
            sublist.append(round(sub_each,2))
    
    _list.append(sublist)

print("Total rows: {}".format(len(_list)))

save_data = {
    'ps': _list,
}

_file = 'dataset/multivariate_samples.txt'

# Save the json file
with open(_file, 'w') as outfile:
    json.dump(save_data, outfile)
'''



# # For drawing purpose, we need to use bivariate example
# data_for_cov = np.array([a,b])
# cov = np.cov(data_for_cov,bias=True)
# mean=np.array([np.mean(a),np.mean(b)])
# distr = scipy.stats.multivariate_normal(cov = cov, mean = mean, seed = 100000)
# data = distr.rvs(size = 500)

# final_data1 = []
# final_data2 = []
# for index in range(len(data)):
#     if data[index][0] <= 0 or data[index][1] <= 0:
#         continue

#     final_data1.append(data[index][0])
#     final_data2.append(data[index][1])



# # Plotting the generated samples
# path_to_save_dir = 'plots/'
# name = path_to_save_dir+"bi_variate"
# plt.plot(final_data1,final_data2, 'o', c='lime',
#              markeredgewidth = 0.5,
#              markeredgecolor = 'black')
# plt.title(f'Covariance between x1 and x2')
# plt.xlabel('param_cut_triggering_dist')
# plt.ylabel('param_adv_to_cut_start_speed')
# plt.axis('equal')
# plt.savefig(name)
# plt.close()


'''
path_to_save_dir = 'plots/'
_file = 'dataset/parameter_space/cutin_space.json'
with open(_file) as f:
    data = json.loads(f.read())

print("total number of scenarios: {}".format(len(data['data']['param_cut_triggering_dist'])))
'''

# Plotting histograms
path_to_save_dir = 'plots/'
_file = 'dataset/parameter_space/cutin_space.json'
with open(_file) as f:
    data = json.loads(f.read())

param_cut_triggering_dist = data['data']['param_cut_triggering_dist']
param_adv_to_cut_start_speed = data['data']['param_adv_to_cut_start_speed']
param_adv_to_cut_start_time = data['data']['param_adv_to_cut_start_time']
param_adv_cutend_speed = data['data']['param_adv_cutend_speed']
param_adv_cut_start_to_end_time = data['data']['param_adv_cut_start_to_end_time']
param_adv_speed_final = data['data']['param_adv_speed_final']
param_adv_cutend_to_scenario_end_time = data['data']['param_adv_cutend_to_scenario_end_time']

a = []
for each in param_cut_triggering_dist:

    if each < 0.75 or each > 30.0:
        continue
    else: 
        a.append(round(each,2))

b = []
for each in param_adv_to_cut_start_speed:
    if each < 2.0 or each > 20.0:
        continue
    else: 
        b.append(round(each,2))

d = []
for each in param_adv_cutend_speed:
    if each < 1.0 or each > 20.0:
        continue
    else: 
        d.append(round(each,2))

f = []
for each in param_adv_speed_final:
    if each < 1.0 or each > 12.0:
        continue
    else: 
        f.append(round(each,2))
    

print("Total rows: {}".format(len(a)))

name = path_to_save_dir+"histo_cut_triggering_dist"
fig_w = 8
fig_h = 4
label_text_size = 25
tick_text_size = 25
kde_color = 'g'
plt.figure(figsize=(fig_w,fig_h))
plt.xlabel("Trigger_dist (m)", size=label_text_size)
plt.ylabel("Frequency", size=label_text_size)
#plt.hist(a,color='g',edgecolor = "white", density=True,bins=15)
sn.distplot(a,bins=20,hist_kws=dict(edgecolor="black", linewidth=1))
plt.tick_params(axis='x', labelsize=tick_text_size)
plt.tick_params(axis='y', labelsize=tick_text_size)
plt.savefig(name,bbox_inches='tight')
plt.close()


name = path_to_save_dir+"histo_adv_to_cut_start_speed"
plt.figure(figsize=(fig_w,fig_h))
plt.xlabel("Cut_in_vel (m/s)", size=label_text_size)
plt.ylabel("Frequency", size=label_text_size)
#plt.hist(b,color='g',edgecolor = "white", density=True,bins=15)
sn.distplot(b,bins=20,hist_kws=dict(edgecolor="black", linewidth=1))
plt.tick_params(axis='x', labelsize=tick_text_size)
plt.tick_params(axis='y', labelsize=tick_text_size)
plt.savefig(name,bbox_inches='tight')
plt.close()

'''
_text = 'param_adv_to_cut_start_speed'
_list = data['data'][_text]
_min = min(_list)
_max = max(_list)
print("min and max of {}: {}-{}".format(_text, _min, _max))
'''
name = path_to_save_dir+"histo_param_adv_to_cut_start_time"
plt.figure(figsize=(fig_w,fig_h))
plt.xlabel("Start_to_cutin_time (s)", size=label_text_size)
plt.ylabel("Frequency", size=label_text_size)
#plt.hist(data['data']['param_adv_to_cut_start_time'])
plt.hist(c,color='g',edgecolor = "white", density=True,bins=15)
plt.tick_params(axis='x', labelsize=tick_text_size)
plt.tick_params(axis='y', labelsize=tick_text_size)
plt.savefig(name,bbox_inches='tight')
plt.close()
'''
_text = 'param_adv_to_cut_start_time'
_list = data['data'][_text]
_min = min(_list)
_max = max(_list)
print("min and max of {}: {}-{}".format(_text, _min, _max))
'''

name = path_to_save_dir+"histo_param_adv_cutend_speed"
plt.figure(figsize=(fig_w,fig_h))
plt.xlabel("Cut_end_vel (m/s)", size=label_text_size)
plt.ylabel("Frequency", size=label_text_size)
#plt.hist(d,color='g',edgecolor = "white", density=True,bins=15)
sn.distplot(d,bins=20,hist_kws=dict(edgecolor="black", linewidth=1))
plt.tick_params(axis='x', labelsize=tick_text_size)
plt.tick_params(axis='y', labelsize=tick_text_size)
plt.savefig(name,bbox_inches='tight')
plt.close()

'''
_text = 'param_adv_cutend_speed'
_list = data['data'][_text]
_min = min(_list)
_max = max(_list)
print("min and max of {}: {}-{}".format(_text, _min, _max))
'''

name = path_to_save_dir+"histo_param_adv_cut_start_to_end_time"
plt.figure(figsize=(fig_w,fig_h))
plt.xlabel("Cutstart_to_cutend_time (s)", size=label_text_size)
plt.ylabel("Frequency", size=label_text_size)
#plt.hist(e, color='g',edgecolor = "white", density=True,bins=15)
sn.distplot(e,bins=20,hist_kws=dict(edgecolor="black", linewidth=1))
plt.tick_params(axis='x', labelsize=tick_text_size)
plt.tick_params(axis='y', labelsize=tick_text_size)
plt.savefig(name,bbox_inches='tight')
plt.close()

'''
_text = 'param_adv_cut_start_to_end_time'
_list = data['data'][_text]
_min = min(_list)
_max = max(_list)
print("min and max of {}: {}-{}".format(_text, _min, _max))
'''

name = path_to_save_dir+"histo_param_adv_speed_final"
plt.figure(figsize=(fig_w,fig_h))
plt.xlabel("Final_vel (m/s)", size=label_text_size)
plt.ylabel("Frequency", size=label_text_size)
plt.hist(f,color='g',edgecolor = "white", density=True,bins=15)
plt.tick_params(axis='x', labelsize=tick_text_size)
plt.tick_params(axis='y', labelsize=tick_text_size)
plt.savefig(name,bbox_inches='tight')
plt.close()

'''
_text = 'param_adv_speed_final'
_list = data['data'][_text]
_min = min(_list)
_max = max(_list)
print("min and max of {}: {}-{}".format(_text, _min, _max))


name = path_to_save_dir+"histo_param_adv_cutend_to_scenario_end_time"
plt.figure(figsize=(fig_w,fig_h))
plt.xlabel("Cutend_to_final_time (s)", size=label_text_size)
plt.ylabel("Frequency", size=label_text_size)
#plt.hist(data['data']['param_adv_cutend_to_scenario_end_time'])
sn.distplot(data['data']['param_adv_cutend_to_scenario_end_time'], color='red', kde_kws={"color": kde_color})
plt.tick_params(axis='x', labelsize=tick_text_size)
plt.tick_params(axis='y', labelsize=tick_text_size)
plt.savefig(name)
plt.close()

_text = 'param_adv_cutend_to_scenario_end_time'
_list = data['data'][_text]
_min = min(_list)
_max = max(_list)
print("min and max of {}: {}-{}".format(_text, _min, _max))


name = path_to_save_dir+"histo_param_ego_speed_init"
plt.figure(figsize=(fig_w,fig_h))
plt.xlabel("Ego_init_vel (m/s)", size=label_text_size)
plt.ylabel("Frequency", size=label_text_size)
#plt.hist(data['data']['param_ego_speed_init'])
sn.distplot(data['data']['param_ego_speed_init'], color='red', kde_kws={"color": kde_color})
plt.tick_params(axis='x', labelsize=tick_text_size)
plt.tick_params(axis='y', labelsize=tick_text_size)
plt.savefig(name)
plt.close()

_text = 'param_ego_speed_init'
_list = data['data'][_text]
_min = min(_list)
_max = max(_list)
print("min and max of {}: {}-{}".format(_text, _min, _max))
'''






