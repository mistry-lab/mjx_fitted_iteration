import mujoco
import numpy
from mujoco import viewer, mj_step


#  desired obj = 0.15, -0.35
# init arm = -1.57, 0 ,0,
# interactive viewer that renders the xml file passed as argument

def interactive_viewer(xml_path: str):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    # set qpos to initial position
    data.qpos = numpy.array([-1.57, 0 ,0, 0.15, -0.35])
    mj_step(model, data)
    with mujoco.viewer.launch(model, data) as v:
        while v.is_running():
            mujoco.mj_step(model, data)
            v.sync()

if __name__ == '__main__':
    interactive_viewer('/home/daniel/Repos/mjx_fitted_iteration/diff_sim/xmls/point_mass_tendon.xml')