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

class JsonFSM:
    def __init__(self) -> None:
        self.state = JsonState.EXPECT_OPEN_BRACE
        self.nesting: list[str] = []
        self.current_buffer: str = ""
        self.escape_next: bool = False

    def feed(self, char: str) -> None:
        if self.state == JsonState.EXPECT_OPEN_BRACE:
            assert char == '{'
            self.nesting.append('{')
            self.state = JsonState.EXPECT_KEY_OR_CLOSE

        elif self.state == JsonState.EXPECT_KEY_OR_CLOSE:
            if char == '"':
                self.state = JsonState.IN_KEY
            elif char == '}':
                self.nesting.pop()
                if not self.nesting:
                    self.state = JsonState.DONE
                else:
                    self.state = JsonState.EXPECT_COMMA_OR_CLOSE
        
        elif self.state == JsonState.IN_KEY:
            if char == '"' and not self.escape_next:
                self.state = JsonState.EXPECT_COLON
                self.escape_next = False
            elif char == '\\':
                self.escape_next = True
            else: 
                self.escape_next = False

        elif self.state == JsonState.EXPECT_COLON:
            assert char == ':'
            self.state = JsonState.EXPECT_VALUE

        elif self.state == JsonState.EXPECT_VALUE:
            if char == '"':
                self.state = JsonState.IN_STRING_VALUE
            elif char in '-0123456789':
                self.state = JsonState.IN_NUMBER_VALUE
            elif char =='{':
                self.nesting.append('{')
                self.state = JsonState.EXPECT_VALUE
        
        elif self.state == JsonState.IN_STRING_VALUE:
            if char == '"' and not self.escape_next:
                self.state = JsonState.EXPECT_COMMA_OR_CLOSE
            elif char == "\\":
                self.escape_next = True
            else:
                self.escape_next = False
            
        elif self.state == JsonState.IN_NUMBER_VALUE:
            if char in ',}]':
                self._handle_delimiter(char)
        
        elif self.state == JsonState.EXPECT_COMMA_OR_CLOSE:
            self._handle_delimiter(char)
    
    def _handle_delimiter(self, char: str) -> None:
        """For handling delimiters"""
        if char == ',':
            if self.nesting[-1] == '{':
                self.state = JsonState.EXPECT_COMMA_OR_CLOSE
            else:
                self.state = JsonState.EXPECT_VALUE
        elif char in '}]':
            self.nesting.pop()
            if not self.nesting:
                self.state = JsonState.DONE
            else:
                self.state = JsonState.EXPECT_COMMA_OR_CLOSE
        
    def allowed_chars(self) -> set[str]:
        if self.state == JsonState.EXPECT_OPEN_BRACE:
            return '{'
        elif self.state == JsonState.EXPECT_KEY_OR_CLOSE:
            return {'"', '}', ' ', '\n', '\t'}
        elif self.state == JsonState.IN_KEY:
            return None

        elif self.state == json
