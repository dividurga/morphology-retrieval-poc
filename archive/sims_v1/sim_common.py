"""rollout interface shared by both sims. same result type, same fall rule."""

from dataclasses import dataclass

FALL_PITCH_THRESH = 1.0  # rad
FALL_HEIGHT_FRAC = 0.5  # of standing torso height


@dataclass
class ResultOfARollout:
    displacement: float
    fell: bool
    fell_time: float | None


def has_fallen(pitch: float, dz: float, standing_height: float) -> bool:
    """dz is the root z offset from its t=0 value."""
    return abs(pitch) > FALL_PITCH_THRESH or standing_height + dz < FALL_HEIGHT_FRAC * standing_height
