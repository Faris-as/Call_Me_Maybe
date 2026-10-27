from enum import Enum, auto
import json
import re


class JsonState(Enum):
    EXPECT_OPEN_BRACE = auto()
    EXPECT_KEY_OR_CLOSE = auto()
    EXPECT_KEY = auto()
    IN_KEY = auto()
    EXPECT_COLON = auto()
    EXPECT_VALUE = auto()
    EXPECT_VALUE_OR_CLOSE = auto()
    IN_STRING_VALUE = auto()
    IN_NUMBER_VALUE = auto()
    IN_LITERAL_VALUE = auto()
    EXPECT_COMMA_OR_CLOSE = auto()
    DONE = auto()


_JSON_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z")
_HEX = set("0123456789abcdefABCDEF")
_ESCAPES = set('"\\/bfnrt')


class JsonFSM:
    """
    Character-by-character validator for a single JSON object.

    This intentionally requires the top-level JSON value to be an object,
    matching the original EXPECT_OPEN_BRACE design.
    """

    def __init__(self) -> None:
        self.state = JsonState.EXPECT_OPEN_BRACE
        self.nesting: list[str] = []
        self.current_buffer = ""
        self.escape_next = False
        self.unicode_digits_remaining = 0
        self.expected_literal = ""
        self.literal_index = 0

    def feed(self, char: str) -> None:
        if len(char) != 1:
            raise ValueError("feed() expects exactly one character")

        # A number can end immediately before whitespace or a delimiter.
        # Validate it, then re-process the current character in the next state.
        if self.state == JsonState.IN_NUMBER_VALUE:
            if char in "0123456789.eE+-":
                self.current_buffer += char
                return
            self._validate_number()
            self.state = JsonState.EXPECT_COMMA_OR_CLOSE
            self.feed(char)
            return

        if self.state == JsonState.IN_LITERAL_VALUE:
            if self.literal_index < len(self.expected_literal):
                if char != self.expected_literal[self.literal_index]:
                    raise ValueError(
                        f"Expected {self.expected_literal[self.literal_index]!r} "
                        f"in literal {self.expected_literal!r}, got {char!r}"
                    )
                self.literal_index += 1
                if self.literal_index == len(self.expected_literal):
                    self.state = JsonState.EXPECT_COMMA_OR_CLOSE
                return

        if self.state == JsonState.EXPECT_OPEN_BRACE:
            if char.isspace():
                return
            if char == "{":
                self.nesting.append("{")
                self.state = JsonState.EXPECT_KEY_OR_CLOSE
            else:
                raise ValueError(f"Expected '{{', got {char!r}")

        elif self.state == JsonState.EXPECT_KEY_OR_CLOSE:
            if char.isspace():
                return
            if char == '"':
                self.current_buffer = ""
                self._reset_string_tracking()
                self.state = JsonState.IN_KEY
            elif char == "}" and self.nesting and self.nesting[-1] == "{":
                self.nesting.pop()
                self.state = JsonState.DONE if not self.nesting else JsonState.EXPECT_COMMA_OR_CLOSE
            else:
                raise ValueError(f"Expected object key or '}}', got {char!r}")

        elif self.state == JsonState.EXPECT_KEY:
            if char.isspace():
                return
            if char == '"':
                self.current_buffer = ""
                self._reset_string_tracking()
                self.state = JsonState.IN_KEY
            else:
                raise ValueError(f"Expected object key, got {char!r}")

        elif self.state == JsonState.IN_KEY:
            if self._consume_string_char(char):
                self.state = JsonState.EXPECT_COLON

        elif self.state == JsonState.EXPECT_COLON:
            if char.isspace():
                return
            if char == ":":
                self.state = JsonState.EXPECT_VALUE
            else:
                raise ValueError(f"Expected ':', got {char!r}")

        elif self.state == JsonState.EXPECT_VALUE_OR_CLOSE:
            if char.isspace():
                return
            if char == "]":
                if not self.nesting or self.nesting[-1] != "[":
                    raise ValueError("Unexpected ']'")
                self.nesting.pop()
                self.state = (
                    JsonState.DONE
                    if not self.nesting
                    else JsonState.EXPECT_COMMA_OR_CLOSE
                )
            else:
                self.state = JsonState.EXPECT_VALUE
                self.feed(char)

        elif self.state == JsonState.EXPECT_VALUE:
            if char.isspace():
                return
            if char == '"':
                self.current_buffer = ""
                self._reset_string_tracking()
                self.state = JsonState.IN_STRING_VALUE
            elif char == "{":
                self.nesting.append("{")
                self.state = JsonState.EXPECT_KEY_OR_CLOSE
            elif char == "[":
                self.nesting.append("[")
                self.state = JsonState.EXPECT_VALUE_OR_CLOSE
            elif char in "-0123456789":
                self.current_buffer = char
                self.state = JsonState.IN_NUMBER_VALUE
            elif char in "tfn":
                self.expected_literal = {"t": "true", "f": "false", "n": "null"}[char]
                self.literal_index = 1
                if len(self.expected_literal) == 1:
                    self.state = JsonState.EXPECT_COMMA_OR_CLOSE
                else:
                    self.state = JsonState.IN_LITERAL_VALUE
            else:
                raise ValueError(f"Expected JSON value, got {char!r}")

        elif self.state == JsonState.IN_STRING_VALUE:
            if self._consume_string_char(char):
                self.state = JsonState.EXPECT_COMMA_OR_CLOSE

        elif self.state == JsonState.EXPECT_COMMA_OR_CLOSE:
            if char.isspace():
                return
            self._handle_delimiter(char)

        elif self.state == JsonState.DONE:
            if not char.isspace():
                raise ValueError(f"JSON already complete; got {char!r}")

    def _reset_string_tracking(self) -> None:
        self.escape_next = False
        self.unicode_digits_remaining = 0

    def _consume_string_char(self, char: str) -> bool:
        """Consume a string character; return True when its closing quote is read."""
        if self.unicode_digits_remaining:
            if char not in _HEX:
                raise ValueError(f"Invalid Unicode escape character: {char!r}")
            self.unicode_digits_remaining -= 1
            self.current_buffer += char
            return False

        if self.escape_next:
            self.escape_next = False
            if char == "u":
                self.unicode_digits_remaining = 4
                self.current_buffer += char
                return False
            if char not in _ESCAPES:
                raise ValueError(f"Invalid JSON escape: \\{char}")
            self.current_buffer += char
            return False

        if char == "\\":
            self.escape_next = True
            self.current_buffer += char
            return False

        if char == '"':
            # Decode the string token to ensure escape syntax is valid.
            try:
                json.loads('"' + self.current_buffer + '"')
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON string: {exc.msg}") from exc
            return True

        if ord(char) < 0x20:
            raise ValueError("Unescaped control character in JSON string")

        self.current_buffer += char
        return False

    def _validate_number(self) -> None:
        if not _JSON_NUMBER.fullmatch(self.current_buffer):
            raise ValueError(f"Invalid JSON number: {self.current_buffer!r}")
        self.current_buffer = ""

    def _handle_delimiter(self, char: str) -> None:
        if not self.nesting:
            raise ValueError(f"Unexpected delimiter {char!r}")

        container = self.nesting[-1]

        if char == ",":
            if container == "{":
                self.state = JsonState.EXPECT_KEY
            else:
                self.state = JsonState.EXPECT_VALUE
            return

        if char == "}" and container == "{":
            self.nesting.pop()
            self.state = (
                JsonState.DONE
                if not self.nesting
                else JsonState.EXPECT_COMMA_OR_CLOSE
            )
            return

        if char == "]" and container == "[":
            self.nesting.pop()
            self.state = (
                JsonState.DONE
                if not self.nesting
                else JsonState.EXPECT_COMMA_OR_CLOSE
            )
            return

        raise ValueError(
            f"Mismatched closing delimiter {char!r} for {container!r}"
        )

    def allowed_chars(self) -> set[str] | None:
        whitespace = {" ", "\n", "\t", "\r"}

        if self.state == JsonState.EXPECT_OPEN_BRACE:
            return {"{"} | whitespace
        if self.state in (JsonState.EXPECT_KEY_OR_CLOSE,):
            return {'"', "}"} | whitespace
        if self.state == JsonState.EXPECT_KEY:
            return {'"'} | whitespace
        if self.state == JsonState.IN_KEY or self.state == JsonState.IN_STRING_VALUE:
            if self.unicode_digits_remaining:
                return _HEX
            if self.escape_next:
                return _ESCAPES | {"u"}
            # JSON strings allow Unicode characters except raw control characters,
            # quotation marks, and backslashes (which must be escaped).
            # Include printable ASCII here; feed() also accepts valid non-ASCII.
            return {chr(codepoint) for codepoint in range(0x20, 0x7F)
                    if chr(codepoint) not in {'"', "\\\\"}}
        if self.state == JsonState.EXPECT_COLON:
            return {":"} | whitespace
        if self.state == JsonState.EXPECT_VALUE_OR_CLOSE:
            return {
                '"', "{", "[", "]", "-", "0", "1", "2", "3", "4",
                "5", "6", "7", "8", "9", "t", "f", "n"
            } | whitespace
        if self.state == JsonState.EXPECT_VALUE:
            return {
                '"', "{", "[", "-", "0", "1", "2", "3", "4",
                "5", "6", "7", "8", "9", "t", "f", "n"
            } | whitespace
        if self.state == JsonState.IN_NUMBER_VALUE:
            return set("0123456789.eE+-") | whitespace | {",", "}", "]"}
        if self.state == JsonState.IN_LITERAL_VALUE:
            return {self.expected_literal[self.literal_index]}
        if self.state == JsonState.EXPECT_COMMA_OR_CLOSE:
            allowed = {","}
            if self.nesting:
                allowed.add("}" if self.nesting[-1] == "{" else "]")
            return allowed | whitespace
        if self.state == JsonState.DONE:
            return whitespace
        return set()

    def finish(self) -> None:
        """Call at end-of-input to reject incomplete JSON."""
        if self.state == JsonState.IN_NUMBER_VALUE:
            self._validate_number()
            self.state = JsonState.EXPECT_COMMA_OR_CLOSE

        if self.state != JsonState.DONE:
            raise ValueError(f"Incomplete JSON; stopped in state {self.state.name}")

    @property
    def is_done(self) -> bool:
        return self.state == JsonState.DONE
