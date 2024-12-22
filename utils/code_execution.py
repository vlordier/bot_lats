import logging
import sys
import io


def extract_and_execute_code(text: str):
    code_start_marker = "```python"
    code_end_marker = "```"
    start = text.find(code_start_marker)
    end = text.find(code_end_marker, start + len(code_start_marker))
    if start != -1 and end != -1:
        code = text[start + len(code_start_marker) : end].strip()
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        try:
            exec(code, globals())
        except Exception as e:
            logging.error(f"Error executing code: {e}, code: {code}")
            sys.stdout = old_stdout
            return f"Error: {e}", code
        sys.stdout = old_stdout
        return new_stdout.getvalue(), code
    return "No code found.", None
