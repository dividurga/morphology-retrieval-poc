"""Build a MuJoCo model from morphology params and roll out the gait controller in it."""

from dataclasses import dataclass

import mujoco
import numpy as np

import controller
import morphology

FALL_PITCH_THRESH = 1.0  # radians (~57 degrees)
FALL_HEIGHT_FRAC = 0.5  # fraction of standing torso height

QPOS_X = 0
QPOS_Z = 1
QPOS_PITCH = 2


@dataclass
class ResultOfARollout:
    displacement: float
    fell: bool
    fell_time: float | None


def make_model(params: dict) -> mujoco.MjModel:
    xml = morphology.build_biped_xml(params)
    model = mujoco.MjModel.from_xml_string(xml)
    actuator_names = [model.actuator(i).name for i in range(model.nu)]
    assert actuator_names == controller.ACTUATOR_ORDER, (
        f"actuator order mismatch: model has {actuator_names}, "
        f"controller.py expects {controller.ACTUATOR_ORDER}"
    )
    return model



def simulate(model: mujoco.MjModel, theta: np.ndarray, duration: float = 8.0) -> ResultOfARollout:
    """Run one open-loop episode and report raw forward displacement.

    On the first step this condition is true, stop simulating and record
    fell=True, fell_time=data.time. Otherwise run the full duration.
    Return the final data.qpos[QPOS_X] as displacement either way.
    """
    data = mujoco.MjData(model)

    while data.time < duration:
        data.ctrl = controller.joint_targets(theta, data.time)
        mujoco.mj_step(model, data)
        if _has_fallen(data, model.body("torso").pos[2]):
            return ResultOfARollout(displacement=data.qpos[QPOS_X], fell=True, fell_time=data.time)

    return ResultOfARollout(displacement=data.qpos[QPOS_X], fell=False, fell_time=None)



def _has_fallen(data: mujoco.MjData, standing_height: float) -> bool:
        """Return True if the robot has fallen, False otherwise."""
        pitch = data.qpos[QPOS_PITCH]
        z = data.qpos[QPOS_Z] + standing_height  # torso height = root z + torso pos z
        # ehhh we might need to change what counts as having fallen but ig for now this is 
        # a good enough start
        return abs(pitch) > FALL_PITCH_THRESH or z < FALL_HEIGHT_FRAC * standing_height

if __name__ == "__main__":
    model = make_model(morphology.NOMINAL_PARAMS)
    result = simulate(model, controller.DEFAULT_THETA, duration=8.0)
    print(result)
