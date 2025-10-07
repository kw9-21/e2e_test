# --- NEW FILE: fca_layer.py ---

import math
import torch
import torch.nn as nn

def get_freq_indices(method):
    """
    Helper function to get frequency coordinates based on a predefined method.
    Copied from the original FcaNet implementation.
    """
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

# class MultiSpectralDCTLayer(nn.Module):
#     """
#     Generate dct filters and apply them to the input.
#     Copied from the original FcaNet implementation.
#     """
#     def __init__(self, height, width, mapper_x, mapper_y, channel):
#         super(MultiSpectralDCTLayer, self).__init__()
        
#         assert len(mapper_x) == len(mapper_y)
#         assert channel % len(mapper_x) == 0

#         self.num_freq = len(mapper_x)
#         self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

#     def forward(self, x):
#         assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
#         x = x * self.weight
#         result = torch.sum(x, dim=[2,3])
#         return result

#     def build_filter(self, pos, freq, POS):
#         result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) 
#         if freq == 0:
#             return result
#         else:
#             return result * math.sqrt(2)
    
#     def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
#         dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)
#         c_part = channel // len(mapper_x)

#         for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
#             for t_x in range(tile_size_x):
#                 for t_y in range(tile_size_y):
#                     dct_filter[i * c_part: (i+1)*c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
                        
#         return dct_filter

class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters and apply them to the input.
    MODIFIED to support non-divisible channel numbers using flexible grouping.
    """
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()
        
        assert len(mapper_x) == len(mapper_y)
        # --- MODIFICATION: REMOVE THE ASSERTION ---
        # assert channel % len(mapper_x) == 0 
        # We will now handle the case where channel is not divisible by num_freq.

        self.num_freq = len(mapper_x)
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

    def forward(self, x):
        # This function does not need to be changed.
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        x = x * self.weight
        result = torch.sum(x, dim=[2,3])
        return result

    def build_filter(self, pos, freq, POS):
        # This function does not need to be changed.
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) 
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)
    
    # --- MODIFICATION: REWRITE get_dct_filter with FLEXIBLE GROUPING ---
    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)
        num_freq = len(mapper_x)

        # Calculate the base number of channels per frequency group
        base_channels_per_freq = channel // num_freq
        # Calculate the number of groups that will get an extra channel
        remainder = channel % num_freq

        current_channel_idx = 0
        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            # Determine the number of channels for this specific frequency group
            channels_in_this_group = base_channels_per_freq
            if i < remainder:
                channels_in_this_group += 1
            
            # If there are no channels in this group, skip (e.g., channel < num_freq)
            if channels_in_this_group == 0:
                continue

            # Generate the DCT filter for this frequency
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    filter_value = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
                    
                    # Assign this filter to the channels in the current group
                    start_channel = current_channel_idx
                    end_channel = current_channel_idx + channels_in_this_group
                    dct_filter[start_channel:end_channel, t_x, t_y] = filter_value
            
            # Move to the next channel index
            current_channel_idx += channels_in_this_group
                        
        return dct_filter