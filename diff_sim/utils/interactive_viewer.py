import mujoco
from mujoco import viewer, mj_step

def interactive_viewer(xml_path: str):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mj_step(model, data)
    with mujoco.viewer.launch(model, data) as v:
        while v.is_running():
            mujoco.mj_step(model, data)
            v.sync()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", help="xml file to render", default="diff_sim/xmls/finger_mjx.xml")
    interactive_viewer(parser.parse_args().xml)

    model = mujoco.MjModel.from_xml_path(parser.parse_args().xml)
    data = mujoco.MjData(model)
