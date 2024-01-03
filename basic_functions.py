import math
import numpy as np

def print_progress_bar(iteration, total, prefix = 'Progress', suffix = 'Complete', decimals = 3, length = 50, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    if iteration == total: 
        print()
        
def _evenDist(total, x, sign = 1, reverse = False):
    output = []
    x = abs(x)
    if x == 0:
        if reverse:
            return [sign for j in range(total)]
        else:
            return [0 for j in range(total)]
    spaceblocks = (total - x)/x

    t = [math.ceil(spaceblocks) if i%2 == 0 else math.floor(spaceblocks) for i in range(x)]

    for i in reversed(range(len(t))): 
        if sum(t) < (total-x) and i%2 == 1:
            t[i] += 1
        if sum(t) > (total-x) and i%2 == 0:
            t[i] -= 1     
    for i in range(x):
        output += [0 if reverse else sign]  + [(sign if reverse else 0) for j in range(t[i])]   
    while len(output) < total:
        output.append(sign if reverse else 0)
    return output[:total]

def evenDist(total, x):
    if abs(x) <= total/2:
        return _evenDist(total, abs(x), sign = np.sign(x), reverse = False)
    else:
        output = _evenDist(total, total - abs(x), sign = np.sign(x), reverse = True)
        output.reverse()
        return output
    
def normalize_dict(d):
    """
    Normalize the values in a dictionary by their maximum absolute value.

    Args:
    d (dict): A dictionary to normalize.

    Returns:
    dict: A new dictionary with the same keys as `d` and normalized values.
    """
    
    if len(d) == 0:
        return d
    # Find absolute maximum value in the dictionary
    max_val = max(abs(val) for val in d.values())

    if max_val == 0:
        return d
    
    # Normalize dictionary by the absolute max value
    normalized_d = {k: v / max_val for k, v in d.items()}
    
    return normalized_d

def value_to_hex(value, a = 0.5):
    
    r,g,b = 1,1,1
    # Piecewise linear interpolation
    if value < 0:  # Blue to White
        value = abs(value)
        r -= value
        g -= value
    else:  # White to Red
        g -= value
        b -= value

    # Convert to 8-bit values
    r_8bit = int(r * 255)
    g_8bit = int(g * 255)
    b_8bit = int(b * 255)
    alpha = int(a * 255)  # Opaque

    # Convert to hex
    hex_code = "#{:02x}{:02x}{:02x}{:02x}".format(r_8bit, g_8bit, b_8bit, alpha)
    return hex_code

def array_to_hex(array, a = 0.5):
    vec_func = np.vectorize(value_to_hex)
    return vec_func(array, a = a)