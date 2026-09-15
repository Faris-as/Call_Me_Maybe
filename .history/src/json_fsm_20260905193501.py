# state machine - FSM - pure logic, no LLM
from enum import Enum, auto

class JsonState(Enum):
    EXPECT_OPEN_BRACE = auto