import os

def save_strings_to_files(directory, **kwargs):
    """
    This function takes a directory path and any number of string variables,
    saving each string to a text file. The filename is based on the name
    of the passed keyword argument.

    Args:
        directory (str): The path to the directory where files will be saved.
        **kwargs: Arbitrary keyword arguments representing the string variables.
    """
    # Ensure the directory exists; if not, create it
    os.makedirs(directory, exist_ok=True)

    for var_name, var_value in kwargs.items():
        if isinstance(var_value, str):
            # Save the string value to a file named after the variable in the specified directory
            file_path = os.path.join(directory, f"{var_name}.txt")
            with open(file_path, "w") as file:
                file.write(var_value)
            print(f"Saved {var_name} to {file_path}")
        
        elif isinstance(var_value, list):
            # If var_value is a list, save each string in the list to separate files
            for index, item in enumerate(var_value):
                if isinstance(item, str):
                    file_path = os.path.join(directory, f"{var_name}_{index}.txt")
                    with open(file_path, "w") as file:
                        file.write(item)
                    print(f"Saved {var_name}_{index} to {file_path}")