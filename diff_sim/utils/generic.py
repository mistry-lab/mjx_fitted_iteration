import time
import equinox as eqx

def save_model(net: eqx.Module, task_name: str):
    """
    Save the model to the disk
    :param net: Equinox module
    :param task_name: str

    To load the model just deserialize it using eqx.tree_deserialise_leaves(path)
    """
    directory = "models" + "_" + task_name
    date_name = time.strftime("%Y%m%d-%H%M%S") + "_" + task_name
    eqx.tree_serialise_leaves(f"./{directory}/{date_name}.eqx", net)
