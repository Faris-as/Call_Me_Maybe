# state machine - FSM - pure logic, no LLM
from enum import Enum, auto

class JsonState(Enum):
    EXPECT_OPEN_BRACE = auto()
    EXPECT_KEY_OR_CLOSE = auto()
    IN_KEY = auto()
    EXPECT_COLON = auto()
    EXPECT_VALUE = auto()
    IN_STRING_VALUE = auto()
    IN_NUMBER_VALUE = auto()
    EXPECT_COMMA_OR_CLOSE = auto()
    DONE = auto()
    
