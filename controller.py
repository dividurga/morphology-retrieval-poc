"""Open-loop sinusoidal gait, shared across all morphologies bcs we want
to see how different morphologies respond to the same gait.

Left and right are independently parameterized -- no more forced mirroring. The old
version hardcoded a +pi phase shift to make the right leg copy the left leg's motion;
that meant two legs of different lengths (now possible, see morphology.py) were still
forced through the identical bias/amplitude/phase, which makes no sense once the body
itself isn't symmetric."""

import numpy as np

ACTUATOR_ORDER = ["hip_left", "knee_left", "hip_right", "knee_right"]

PARAM_NAMES = [
    "omega",
    "b_hip_left", "a_hip_left", "phi_hip_left",
    "b_knee_left", "a_knee_left", "phi_knee_left",
    "b_hip_right", "a_hip_right", "phi_hip_right",
    "b_knee_right", "a_knee_right", "phi_knee_right",
]
N_PARAMS = len(PARAM_NAMES)

_HIP_BOUNDS = [(-0.6, 0.6), (0.0, 0.8), (0.0, 2 * np.pi)]   # b_hip, a_hip, phi_hip
_KNEE_BOUNDS = [(-1.8, -0.2), (0.0, 0.9), (0.0, 2 * np.pi)]  # b_knee, a_knee, phi_knee

THETA_BOUNDS = [(2.0, 8.0)] + _HIP_BOUNDS + _KNEE_BOUNDS + _HIP_BOUNDS + _KNEE_BOUNDS

# Default starting guess: bias/amplitude equal left==right (no reason to prefer one side
# at init), phase offset by pi on the right so the DEFAULT gait still alternates legs
# (a walking gait, not both legs swinging in sync) -- this reproduces the old formula's
# output exactly when left and right ARE symmetric, so it's a true generalization, not a
# behavior change, for every morphology that happens to be symmetric.
_B_HIP, _A_HIP, _PHI_HIP = 0.1, 0.4, 0.0
_B_KNEE, _A_KNEE, _PHI_KNEE = -0.9, 0.5, np.pi / 2
DEFAULT_THETA = np.array([
    4.5,
    _B_HIP, _A_HIP, _PHI_HIP, _B_KNEE, _A_KNEE, _PHI_KNEE,
    _B_HIP, _A_HIP, _PHI_HIP + np.pi, _B_KNEE, _A_KNEE, _PHI_KNEE + np.pi,
])


def theta_bounds(omega_scale: float = 1.0) -> list:
    """THETA_BOUNDS with omega scaled to the robot's time scale (spec.omega_scale)."""
    b = list(THETA_BOUNDS)
    b[0] = (b[0][0] * omega_scale, b[0][1] * omega_scale)
    return b


def default_theta(omega_scale: float = 1.0) -> np.ndarray:
    theta = DEFAULT_THETA.copy()
    theta[0] *= omega_scale
    return theta


def joint_targets(theta: np.ndarray, t: float) -> np.ndarray:
    """Return target angles [hip_left, knee_left, hip_right, knee_right] at time t.

    theta unpacks per PARAM_NAMES: one shared omega, then independent
    (bias, amplitude, phase) for hip_left, knee_left, hip_right, knee_right in turn.
    Return order must match ACTUATOR_ORDER / the actuator order in morphology.py.
    """
    (omega,
     b_hip_left, a_hip_left, phi_hip_left,
     b_knee_left, a_knee_left, phi_knee_left,
     b_hip_right, a_hip_right, phi_hip_right,
     b_knee_right, a_knee_right, phi_knee_right) = theta
    hip_left_at_t = b_hip_left + a_hip_left * np.sin(omega * t + phi_hip_left)
    knee_left_at_t = b_knee_left + a_knee_left * np.sin(omega * t + phi_knee_left)
    hip_right_at_t = b_hip_right + a_hip_right * np.sin(omega * t + phi_hip_right)
    knee_right_at_t = b_knee_right + a_knee_right * np.sin(omega * t + phi_knee_right)
    return np.array([hip_left_at_t, knee_left_at_t, hip_right_at_t, knee_right_at_t])


if __name__ == "__main__":
    for t in np.linspace(0, 2.0, 5):
        q = joint_targets(DEFAULT_THETA, t)
        assert q.shape == (4,)
        print(f"t={t:.2f}  hip_left={q[0]:+.3f} knee_left={q[1]:+.3f} hip_right={q[2]:+.3f} knee_right={q[3]:+.3f}")
