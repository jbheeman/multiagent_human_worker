import subprocess
from smolagents import tool

@tool
def execute_file(file_path: str) -> str:
    """
    Executes a file and returns its output.

    Args:
        file_path: The path to the file to execute.

    Returns:
        The output of the executed file.
    """
    try:
        result = subprocess.run(
            ["python", file_path],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error executing file: {e.stderr}"
    except FileNotFoundError:
        return f"Error: File not found at {file_path}"
