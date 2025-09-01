import json
import os
from .logger import log  # Assuming you have a configured logger instance named 'log'


def save_json(data, filename: str, indent: int = 4, dir: str = "test_data"):
    """
    Saves a Python object as a JSON file using structured logging.

    Args:
        data: The Python object (e.g., dictionary, list) to be saved.
        filename: The path to the file where the JSON data will be written.
        indent: The number of spaces for indentation.
        dir: The directory where the file will be saved.
    """
    fp = os.path.join(dir, filename)
    try:
        # Ensure the directory exists
        os.makedirs(dir, exist_ok=True)

        with open(fp, "w") as f:
            json.dump(data, f, indent=indent)

        # Use log.info for successful, routine operations
        log.info(f"Data successfully saved to {fp}")

    except IOError as e:
        # Use log.error for I/O problems that prevent the operation
        log.error(f"Failed to save data to {fp} due to an I/O error.", exc_info=True)

    except TypeError as e:
        # Use log.error for data-related problems that prevent the operation
        log.error(f"Data could not be serialized to JSON for file {fp}.", exc_info=True)

    except Exception as e:
        # A general catch-all for any other unexpected errors
        log.error(
            f"An unexpected error occurred while saving JSON to {fp}.", exc_info=True
        )


def write_to_file(filename: str, content: any, dir: str = "test_data"):
    """Writes the given content to a specified file using structured logging."""
    fp = os.path.join(dir, filename)
    try:
        # Ensure the directory exists
        os.makedirs(dir, exist_ok=True)

        with open(fp, "w") as f:
            if not isinstance(content, str):
                # Use log.debug for internal logic details that are useful for developers
                log.debug(
                    f"Content for {fp} is not a string, converting from type {type(content).__name__}."
                )
                content = str(content)
            f.write(content)

        # Use log.info for successful, routine operations
        log.info(f"Data successfully written to {fp}")

    except IOError as e:
        # Use log.error for I/O problems
        log.error(f"Failed to write to file {fp} due to an I/O error.", exc_info=True)

    except Exception as e:
        # A general catch-all
        log.error(
            f"An unexpected error occurred while writing to file {fp}.", exc_info=True
        )


def spy_on_chain(x: any, step_name: str = "Unnamed Step"):
    """
    A "spy" function that uses debug-level logging to inspect data in a chain.
    It logs the input it receives and returns it unchanged.
    """
    # log.debug is perfect for this, as it's verbose and only needed during development.
    # It can be disabled in production by setting the logger level to INFO or higher.
    log.debug(f"--- SPY on '{step_name}' ---")

    # Pretty-print dictionaries or lists for better readability in logs
    if isinstance(x, (dict, list)):
        try:
            formatted_data = json.dumps(
                x, indent=2, default=str
            )  # Use default=str for non-serializable objects
            log.debug(formatted_data)
        except TypeError:
            log.debug(str(x))  # Fallback to string conversion
    else:
        log.debug(x)

    log.debug(f"--- END SPY on '{step_name}' ---\n")
    return x
