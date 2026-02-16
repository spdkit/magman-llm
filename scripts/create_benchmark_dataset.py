import os
import glob
import re

def parse_oszicar(filepath):
    """
    Parses an OSZICAR file to extract the number of electronic steps
    and the final energy change (dE).

    Args:
        filepath (str): The path to the OSZICAR file.

    Returns:
        tuple: A tuple containing (number_of_steps, final_dE).
               Returns (None, None) if parsing fails.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if not lines:
            return None, None

        last_line = lines[-1].strip()
        
        # Regex to find "d E =" and capture the following number from the summary line
        match = re.search(r"d E\s*=\s*([-\.0-9E\+]+)", last_line)
        if not match:
            # If dE is not in the last line, it's likely not a proper summary line
            return None, None
            
        final_dE = float(match.group(1))

        # Find the step number from the line before the summary line
        num_steps = 0
        if len(lines) > 1:
            step_line = lines[-2].strip()
            step_match = re.match(r"^(?:RMM|CG|DAV):\s*(\d+)", step_line)
            if step_match:
                num_steps = int(step_match.group(1))
            else: # If the second to last line is not a step line, maybe the job failed early
                return None, None
        else:
            return None, None # Not enough lines to be a valid file

        return num_steps, final_dE

    except (IOError, IndexError, ValueError) as e:
        # print(f"Warning: Could not parse file {filepath}. Reason: {e}")
        return None, None

def create_dataset():
    """
    Generates a CSV dataset of VASP calculations labeled as converged (1)
    or not converged (0), including step count information.
    """
    # Go up one level from scripts to the project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(project_root, '..', 'vasp-data-files')
    output_csv = os.path.join(project_root, '..', 'benchmark_dataset.csv')
    
    search_path = os.path.join(root_dir, '**', 'OSZICAR')
    all_oszicar_files = glob.glob(search_path, recursive=True)
    
    print(f"Found {len(all_oszicar_files)} OSZICAR files to process in {root_dir}.")

    labeled_data = []
    for filepath in all_oszicar_files:
        num_steps, final_dE = parse_oszicar(filepath)

        if num_steps is None or final_dE is None:
            continue

        label = None
        # Rule for "Not Converged" - reached maximum steps (200)
        if num_steps >= 200:
            label = 0
        # Rule for "Converged" - completed before reaching maximum steps
        elif num_steps < 200:
            label = 1

        if label is not None:
            # Make path relative to the project root for cleaner output
            relative_path = os.path.relpath(filepath, os.path.join(project_root, '..'))
            labeled_data.append(f"{relative_path},{label},{num_steps},{final_dE}")

    with open(output_csv, 'w') as f_out:
        f_out.write("filepath,label,num_steps,final_dE\n")
        f_out.write("\n".join(labeled_data))

    print(f"Finished. Wrote {len(labeled_data)} labeled entries to {output_csv}.")

if __name__ == "__main__":
    create_dataset()
