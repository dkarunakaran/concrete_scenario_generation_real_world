# This class is for search space i.e parameter space

from collections import OrderedDict
import numpy as np
import tensorflow as tf
import rospy

class SearchSpace:

    def __init__(self, ps_data=None):
        self.parameters = OrderedDict()
        self.parameter_count_ = 0
        self.value_count_ = 0
        self.feature_count = 0
        self.ps_data = ps_data
        self.con_settings = rospy.get_param('nas_controller')
    
    # our states/actions are represented using parameters
    def add_parameters(self, name = None, values = None):
        index_map = {}
        for i, val in enumerate(values):
            index_map[i] = val
            self.feature_count += 1
            self.value_count_ += 1

        value_map = {}
        for i, val in enumerate(values):
            value_map[val] = i

        # # Newly added
        # value_map_further = {}
        # for i, val in enumerate(values):
        #     val_expanded = self.get_right_value(val)
        #     joined = '_'.join(map(str,val_expanded))
        #     value_map_further[joined] = i

        metadata = {
            'id': self.parameter_count_,
            'name': name,
            'values': values,
            'size': len(values),
            'index_map_': index_map,
            'value_map_': value_map
            #'value_map_further_': value_map_further
        }
        self.parameters[self.parameter_count_] = metadata
        self.parameter_count_ += 1

    def get_parameter_value(self, id, index):
        '''
        Retrieves the parameter value from the parameter value ID
        Args:
            id: global id of the parameter
            index: index of the parameter value (usually from argmax)
        Returns:
            The actual parameter value at given value index
        '''
        parameter = self[id]
        index_map = parameter['index_map_']

        if (type(index) == list or type(index) == np.ndarray) and len(index) == 1:
            index = index[0]

        value = index_map[index]
        return round(value, 3)

    def get_hot_encoded_target(self, state):

        target = []
        state = tf.keras.backend.eval(state)
        for index in range(state.shape[2]):
            parameter_item = int(state[0][0][index].item())
            parameter = self[index]
            value_map = parameter['value_map_']
            value_idx = value_map[parameter_item]
            target_zeros = np.zeros(len(value_map), dtype=np.float32)
            for each in range(target_zeros.shape[0]):
                if each == value_idx:
                    target_zeros[each] = 1
                    target.append(1)
                else:
                    target.append(0)
            
        return target

    # def embedding_encode(self, id, value):
    #     '''
    #     Embedding index encode the specific state value
    #     Args:
    #         id: global id of the state
    #         value: state value
    #     Returns:
    #         embedding encoded representation of the state value
    #     '''

    #     state = self[id]
    #     size = state['size']
    #     value_map = state['value_map_further_']
    #     value = value[0][0].tolist()
    #     updated_val = [round(num, 2) for num in value]
    #     value_idx = value_map['_'.join(map(str,updated_val))]
    #     one_hot = np.zeros((1, size), dtype=np.float32)
    #     one_hot[np.arange(1), value_idx] = 1
    #     return one_hot

    
    def embedding_encode(self, id, value):
        '''
        Embedding index encode the specific state value
        Args:
            id: global id of the state
            value: state value
        Returns:
            embedding encoded representation of the state value
        '''

        state = self[id]
        size = state['size']
        value_map = state['value_map_']
        value_idx = value_map[value]
        one_hot = np.zeros((1, size), dtype=np.float32)
        one_hot[np.arange(1), value_idx] = 1
        return one_hot

    def get_parameter_size(self, id):
        state = self[id]
        size = state['size']
        
        return size


    # def get_random_parameter(self, _type=None):
    #     '''
    #     Constructs a random initial parameter space for feeding as an initial value
    #     to the Controller RNN
    #     Args:
    #         num_layers: number of layers to duplicate the search space
    #     Returns:
    #         A list of one hot encoded parameter
    #     '''

    #     parameters = None

    #     if _type == 'new':
    #         parameter = self[0]
    #         size = parameter['size']
    #         sample = np.random.choice(size, size=1)
    #         sample = parameter['index_map_'][sample[0]]
    #         sample = self.get_right_value(sample)
    #         parameters = np.array(sample, dtype=np.float32)
    #     else:
    #         for id in range(self.size):
    #             parameter = self[id]
    #             size = parameter['size']
    #             sample = np.random.choice(size, size=1)
    #             sample = parameter['index_map_'][sample[0]]
    #             value_map = parameter['value_map_']
    #             value_idx = value_map[sample]
    #             if parameters is None:
    #                 parameters = np.array(round(sample, 3), dtype=np.float32)
    #             else:
    #                 parameters = np.column_stack((parameters, np.array(round(sample, 3), dtype=np.float32)))
                
    #     return parameters

    def get_random_parameter(self):
        '''
        Constructs a random initial parameter space for feeding as an initial value
        to the Controller RNN
        Args:
            num_layers: number of layers to duplicate the search space
        Returns:
            A list of one hot encoded parameter
        '''

        size = len(self.ps_data)
        sample = np.random.choice(size, size=1)
        sample = self.get_right_value(sample[0])
        parameters = np.array(sample, dtype=np.float32)
                
        return parameters

    def get_right_value(self, index):
        
        return self.ps_data[index]

    
    def __getitem__(self, id):
        return self.parameters[id % self.size]

    @property
    def size(self):
        return self.parameter_count_

    @property
    def value_size(self):
        return self.value_count_
    



if __name__ == "__main__":
    SearchSpace()