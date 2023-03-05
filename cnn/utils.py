import numpy as np
from config_cnn import mid_window, mid_step



def split_mel_spec(mel):
    start = 0
    segment_length = mid_window * 50
    step_length = mid_step * 50
    seq_length = mel.shape[1]
    segments = []
    while start < seq_length:
        if start + segment_length > seq_length:
            fill_data = seq_length - start
            empty_data = segment_length - fill_data
            segment = np.pad(mel[:, start:], [(0, 0), (0, empty_data)], mode='constant', constant_values=0)
        else:
            segment = mel[:, start:start + segment_length]
        start += step_length
        segments.append(segment)
    return segments
