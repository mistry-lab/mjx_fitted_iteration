import time
import equinox as eqx
import os

def save_model(net: eqx.Module, directory: str, task_name: str):
    """
    Save the model to the disk with filename as task_name + current time under the models directory.
    :param directory:
    :param net: Equinox module
    :param task_name: str

    To load the model just deserialize it using eqx.tree_deserialise_leaves(path)
    """
    directory = "models" + "/" + directory
    os.makedirs(directory, exist_ok=True)
    date_name = task_name + "_" + time.strftime("%Y%m%d-%H%M%S")
    eqx.tree_serialise_leaves(f"./{directory}/{date_name}.eqx", net)
