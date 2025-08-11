import random
from enum import Enum
from typing import List


class LivenessAction(Enum):
    """
    An enumeration of possible liveness detection actions.
    The value is the user-facing instruction.
    """
    BLINK_TWICE = "2回まばたきしてください"
    TURN_LEFT = "顔を左に向けてください"
    TURN_RIGHT = "顔を右に向けてください"
    OPEN_MOUTH = "口を開けてください"
    NOD = "頷いてください"


class ChallengeGenerator:
    """
    Generates a random sequence of liveness challenges.
    """

    def __init__(self, actions: List[LivenessAction] = None):
        """
        Initializes the generator with a list of possible actions.
        Args:
            actions: A list of LivenessAction enums. Defaults to all actions.
        """
        self.actions = actions if actions else list(LivenessAction)

    def generate(self, num_steps: int = 0) -> List[LivenessAction]:
        """
        Generates a random challenge sequence.

        Args:
            num_steps: The number of steps in the challenge.
                       If 0, it will be a random number between 2 and 3.

        Returns:
            A list of LivenessAction enums representing the challenge sequence.
        """
        if num_steps == 0:
            # As per requirements, generate a sequence of 2 or 3 steps.
            num_steps = random.randint(2, 3)

        # Ensure the number of steps does not exceed the number of available unique actions.
        if num_steps > len(self.actions):
            raise ValueError("Number of steps cannot be greater than the number of available actions.")

        return random.sample(self.actions, num_steps)
