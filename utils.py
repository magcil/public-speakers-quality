import os
import pickle
import time



def save_model(model_dict, out_folder, out_model=None, name=None):
    """
    Saves a model dictionary
    :param model_dict: model dictionary
    :param out_model: path to save the model (optional)
    :param name: name of the model (optional)
    :param is_text: True if the model is a text classifier
                    False if it is an audio classifier
    """
    script_dir = os.path.dirname(__file__)

    if out_model is None:
        timestamp = time.ctime()
        out_model = "{}_{}.pt".format(name, timestamp)
        out_model = out_model.replace(' ', '_')
    else:
        out_model = str(out_model)
        if '.pt' not in out_model or '.pkl' not in out_model:
            out_model = ''.join([out_model, '.pt'])

    if not script_dir:
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        out_path = os.path.join(out_folder, out_model)
    else:
        out_folder = os.path.join(script_dir, out_folder)
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        out_path = os.path.join(out_folder, out_model)

    print(f"\nSaving model to: {out_path}\n")
    with open(out_path, 'wb') as handle:
        pickle.dump(model_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
