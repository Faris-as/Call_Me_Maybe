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
        if len(char) != 1:
            raise ValueError("feed() expects exactly one character")
        
        if self.state == JsonState.EXPECT_OPEN_BRACE:
            if char == "{":
                self.nesting.append('{')
                self.state = JsonState.EXPECT_KEY_OR_CLOSE
        elif not char.isspace():
            raise ValueError(f"Expected '{{', got {char!r}")
        
        elif self.state == JsonState.EXPECT_KEY_OR_CLOSE:
            if char.isspace():
                return
            if char == '"':
                self.state = JsonState.IN_KEY
                self.escape_next = False
            elif char == '}' and self.nesting and self.nesting[-1] == "{":
                self.nesting.pop()
                self.state = JsonState.DONE if not self.nesting else JsonState.EXPECT_COMMA_OR_CLOSE
            else:
                raise ValueError(f"Expected object key or '}}', got {char!r}")
        
        elif self.state == JsonState.IN_KEY:
            if self.escape_next:
                self.escape_next = False
            elif char == "\\":
                self.escape_next = True
            elif char == '"':
                self.state = JsonState.EXPECT_COLON

        elif self.state == JsonState.EXPECT_COLON:
            if char.isspace():
                return
            if char == ":":
                self.state = JsonState.EXPECT_VALUE
            else:
                raise ValueError(f"Expected ':', got {char!r}")

        elif self.state == JsonState.EXPECT_VALUE:
            if char.isspace():
                return
            if char == '"':
                self.state = JsonState.IN_STRING_VALUE
                self.escape_next = False
            elif char == "{":
                self.nesting.append("{")
                self.state = JsonState.EXPECT_KEY_OR_CLOSE
            elif char == "[":
                self.nesting.append("[")
                self.state = JsonState.EXPECT_VALUE
            elif char in "-0123456789":
                self.number_buffer = char
                self.state = JsonState.IN_NUMBER_VALUE
            elif char in "tfn":
                # Literal validation is outside this first FSM version.
                self.state = JsonState.EXPECT_COMMA_OR_CLOSE
            else:
                raise ValueError(f"Expected JSON value, got {char!r}")

        elif self.state == JsonState.IN_STRING_VALUE:
            if self.escape_next:
                self.escape_next = False
            elif char == "\\":
                self.escape_next = True
            elif char == '"':
                self.state = JsonState.EXPECT_COMMA_OR_CLOSE

        elif self.state == JsonState.IN_NUMBER_VALUE:
            if char in "0123456789.eE+-":
                self.number_buffer += char
            elif char.isspace():
                return
            elif char in ",}]":
                self.number_buffer = ""
                self._handle_delimiter(char)
            else:
                raise ValueError(f"Invalid character in number: {char!r}")

        elif self.state == JsonState.EXPECT_COMMA_OR_CLOSE:
            if char.isspace():
                return
            self._handle_delimiter(char)

        elif self.state == JsonState.DONE:
            if not char.isspace():
                raise ValueError(f"JSON already complete; got {char!r}")
    
    def _handle_delimiter(self, char: str) -> None:
        """For handling delimiters"""
        if not nesting:
            raise ValueError(f"Unexpected delimiter {char!r}")
        if char == ',':
            if self.nesting[-1] == '{':
                self.state = JsonState.EXPECT_COMMA_OR_CLOSE
            else:
                self.state = JsonState.EXPECT_VALUE
            return

        if char == "}" and self.nesting[-1] == "{":
            self.nesting.pop()
            self.state = JsonState.DONE if not self.nesting else JsonState.EXPECT_COMMA_OR_CLOSE
            return

        if char == "]" and container == "[":
            self.nesting.pop()
            self.state = JsonState.DONE if not self.nesting else JsonState.EXPECT_COMMA_OR_CLOSE
            return

        raise ValueError(
            f"Mismatched closing delimiter {char!r} for {container!r}"
        )
        
    def allowed_chars(self) -> set[str] | None:
        whitespace = {" ", "\n", "\t", "\r"}

        if self.state == JsonState.EXPECT_OPEN_BRACE:
            return {"{"}

        if self.state == JsonState.EXPECT_KEY_OR_CLOSE:
            return {'"', "}"} | whitespace

        if self.state == JsonState.IN_KEY:
            return None

        if self.state == JsonState.EXPECT_COLON:
            return {":"} | whitespace

        if self.state == JsonState.EXPECT_VALUE:
            return {
                '"', "{", "[", "-", "0", "1", "2", "3", "4",
                "5", "6", "7", "8", "9", "t", "f", "n"
            } | whitespace

        if self.state == JsonState.IN_STRING_VALUE:
            return None

        if self.state == JsonState.IN_NUMBER_VALUE:
            return {
                "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                ".", "e", "E", "+", "-", ",", "}", "]"
            } | whitespace

        if self.state == JsonState.EXPECT_COMMA_OR_CLOSE:
            allowed = {","}
            if self.nesting:
                allowed.add("}" if self.nesting[-1] == "{" else "]")
            return allowed | whitespace

        if self.state == JsonState.DONE:
            return set()

        return set()
    
    @property
    def is_done(self) -> bool:
        return self.state == JsonState.DONE

