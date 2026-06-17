"""
Tests for nanochat.execution - sandboxed code execution utilities.

Run: python -m pytest tests/test_execution.py -v
"""

import pytest
from nanochat.execution import (
    ExecutionResult,
    execute_code,
    WriteOnlyStringIO,
    chdir,
)


class TestExecutionResult:
    """Test ExecutionResult dataclass."""

    def test_success_repr(self):
        r = ExecutionResult(success=True, stdout="hello\n", stderr="")
        assert "success=True" in repr(r)
        assert "hello" in repr(r)

    def test_timeout_repr(self):
        r = ExecutionResult(success=False, stdout="", stderr="", timeout=True, error="Timed out")
        assert "timeout=True" in repr(r)
        assert "Timed out" in repr(r)

    def test_memory_exceeded_repr(self):
        r = ExecutionResult(success=False, stdout="", stderr="", memory_exceeded=True, error="OOM")
        assert "memory_exceeded=True" in repr(r)


class TestExecuteCode:
    """Test execute_code function."""

    def test_simple_print(self):
        result = execute_code("print('hello world')")
        assert result.success is True
        assert result.stdout == "hello world\n"
        assert result.stderr == ""
        assert result.error is None

    def test_multiline_code(self):
        code = "x = 2 + 3\nprint(x)"
        result = execute_code(code)
        assert result.success is True
        assert result.stdout.strip() == "5"

    def test_syntax_error(self):
        result = execute_code("def foo(")
        assert result.success is False
        assert result.error is not None
        assert "SyntaxError" in result.error

    def test_runtime_error(self):
        result = execute_code("1 / 0")
        assert result.success is False
        assert "ZeroDivisionError" in result.error

    def test_name_error(self):
        result = execute_code("print(undefined_variable)")
        assert result.success is False
        assert "NameError" in result.error

    def test_timeout(self):
        code = "import time; time.sleep(10)"
        result = execute_code(code, timeout=1.0)
        assert result.success is False
        assert result.timeout is True

    def test_import_standard_library(self):
        code = "import math; print(math.pi)"
        result = execute_code(code)
        assert result.success is True
        assert "3.14159" in result.stdout

    def test_stderr_capture(self):
        code = "import sys; print('err', file=sys.stderr)"
        result = execute_code(code)
        assert result.success is True
        assert "err" in result.stderr

    def test_empty_code(self):
        result = execute_code("")
        assert result.success is True
        assert result.stdout == ""

    def test_multiple_prints(self):
        code = "print('a')\nprint('b')\nprint('c')"
        result = execute_code(code)
        assert result.success is True
        assert result.stdout == "a\nb\nc\n"

    def test_list_comprehension(self):
        code = "print([x**2 for x in range(5)])"
        result = execute_code(code)
        assert result.success is True
        assert "[0, 1, 4, 9, 16]" in result.stdout

    def test_class_definition(self):
        code = """
class Dog:
    def __init__(self, name):
        self.name = name
    def bark(self):
        return f"{self.name} says woof"

d = Dog("Rex")
print(d.bark())
"""
        result = execute_code(code)
        assert result.success is True
        assert "Rex says woof" in result.stdout


class TestWriteOnlyStringIO:
    """Test WriteOnlyStringIO."""

    def test_write_works(self):
        sio = WriteOnlyStringIO()
        sio.write("hello")
        # getvalue still works because we only override read methods
        assert sio.getvalue() == "hello"

    def test_read_raises(self):
        sio = WriteOnlyStringIO()
        with pytest.raises(IOError):
            sio.read()

    def test_readline_raises(self):
        sio = WriteOnlyStringIO()
        with pytest.raises(IOError):
            sio.readline()

    def test_readlines_raises(self):
        sio = WriteOnlyStringIO()
        with pytest.raises(IOError):
            sio.readlines()

    def test_readable_returns_false(self):
        sio = WriteOnlyStringIO()
        assert sio.readable() is False


class TestChdir:
    """Test chdir context manager."""

    def test_chdir_dot_is_noop(self):
        import os
        before = os.getcwd()
        with chdir("."):
            during = os.getcwd()
        after = os.getcwd()
        assert before == during == after

    def test_chdir_and_restore(self):
        import os
        import tempfile
        before = os.getcwd()
        with tempfile.TemporaryDirectory() as td:
            with chdir(td):
                during = os.getcwd()
                assert during == td
            after = os.getcwd()
            assert after == before
