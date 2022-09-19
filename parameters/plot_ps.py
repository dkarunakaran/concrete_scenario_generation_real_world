import os
import matplotlib

# For the error: Exception ignored in: <bound method Image.del of <tkinter.PhotoImage object at 0x7f1b5f86a710>> Traceback (most recent call last):
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import json
import math
import numpy as np

#-----------------------------------esmini data------------------------------
path_to_save_dir = 'plots/'
_file = 'dataset/parameter_space/cutout_space.json'
#_file = 'dataset/parameter_space/cutin_space.json'
with open(_file) as f:
    data = json.loads(f.read())

print("total number of scenarios: {}".format(len(data['data']['param_cut_triggering_dist'])))


name = path_to_save_dir+"histo_cut_triggering_dist"
plt.figure()
plt.xlabel("triggering_dist(m)")
plt.ylabel("frequency")
plt.hist(data['data']['param_cut_triggering_dist'])
plt.savefig(name)
plt.close()

_text = 'param_cut_triggering_dist'
_list = data['data'][_text]
_min = min(_list)
_max = max(_list)
print("min and max of {}: {}-{}".format(_text, _min, _max))

name = path_to_save_dir+"histo_adv_to_cut_start_speed"
plt.figure()
plt.xlabel("adv_to_cut_start_speed(m/s)")
plt.ylabel("frequency")
plt.hist(data['data']['param_adv_to_cut_start_speed'])
plt.savefig(name)
plt.close()

_text = 'param_adv_to_cut_start_speed'
_list = data['data'][_text]
_min = min(_list)
_max = max(_list)
print("min and max of {}: {}-{}".format(_text, _min, _max))

name = path_to_save_dir+"histo_param_adv_to_cut_start_time"
plt.figure()
plt.xlabel("adv_to_cut_start_time(s)")
plt.ylabel("frequency")
plt.hist(data['data']['param_adv_to_cut_start_time'])
plt.savefig(name)
plt.close()

_text = 'param_adv_to_cut_start_time'
_list = data['data'][_text]
_min = min(_list)
_max = max(_list)
print("min and max of {}: {}-{}".format(_text, _min, _max))


name = path_to_save_dir+"histo_param_adv_cutend_speed"
plt.figure()
plt.xlabel("adv_cutend_speed(m/s)")
plt.ylabel("frequency")
plt.hist(data['data']['param_adv_cutend_speed'])
plt.savefig(name)
plt.close()

_text = 'param_adv_cutend_speed'
_list = data['data'][_text]
_min = min(_list)
_max = max(_list)
print("min and max of {}: {}-{}".format(_text, _min, _max))



name = path_to_save_dir+"histo_param_adv_cut_start_to_end_time"
plt.figure()
plt.xlabel("adv_cut_start_to_end_time(s)")
plt.ylabel("frequency")
plt.hist(data['data']['param_adv_cut_start_to_end_time'])
plt.savefig(name)
plt.close()

_text = 'param_adv_cut_start_to_end_time'
_list = data['data'][_text]
_min = min(_list)
_max = max(_list)
print("min and max of {}: {}-{}".format(_text, _min, _max))


name = path_to_save_dir+"histo_param_adv_speed_final"
plt.figure()
plt.xlabel("adv_speed_final(m/s)")
plt.ylabel("frequency")
plt.hist(data['data']['param_adv_speed_final'])
plt.savefig(name)
plt.close()

_text = 'param_adv_speed_final'
_list = data['data'][_text]
_min = min(_list)
_max = max(_list)
print("min and max of {}: {}-{}".format(_text, _min, _max))


name = path_to_save_dir+"histo_param_adv_cutend_to_scenario_end_time"
plt.figure()
plt.xlabel("adv_cutend_to_scenario_end_time(s)")
plt.ylabel("frequency")
plt.hist(data['data']['param_adv_cutend_to_scenario_end_time'])
plt.savefig(name)
plt.close()

_text = 'param_adv_cutend_to_scenario_end_time'
_list = data['data'][_text]
_min = min(_list)
_max = max(_list)
print("min and max of {}: {}-{}".format(_text, _min, _max))


name = path_to_save_dir+"histo_param_ego_speed_init"
plt.figure()
plt.xlabel("ego_speed_init(m/s)")
plt.ylabel("frequency")
plt.hist(data['data']['param_ego_speed_init'])
plt.savefig(name)
plt.close()

_text = 'param_ego_speed_init'
_list = data['data'][_text]
_min = min(_list)
_max = max(_list)
print("min and max of {}: {}-{}".format(_text, _min, _max))


