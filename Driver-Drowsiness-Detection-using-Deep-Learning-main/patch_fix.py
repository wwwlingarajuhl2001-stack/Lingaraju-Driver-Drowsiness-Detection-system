import os
import site

# Define the specific file path based on your error log
target_file = r"D:\New folder\Driver-Drowsiness-Detection-using-Deep-Learning-main\drow_env\Lib\site-packages\keras\src\legacy\saving\saving_utils.py"

if os.path.exists(target_file):
    with open(target_file, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    fixed_count = 0
    
    for line in lines:
        # This replaces any .pop("something", None) with a Python 3.13 safe version
        if '.pop(' in line and ', None)' in line:
            # Extract the key name (e.g., "axis" or "batch_input_shape")
            parts = line.split('.pop(')
            prefix = parts[0]
            suffix = parts[1].split(', None)')[0]
            remainder = parts[1].split(', None)')[1]
            
            # Reconstruct the line: dict.pop(key) if key in dict else None
            # We assume the dictionary name is the last word before .pop
            dict_name = prefix.split()[-1]
            new_line = f"{prefix}.pop({suffix}) if {suffix} in {dict_name} else None{remainder}"
            new_lines.append(new_line)
            fixed_count += 1
        else:
            new_lines.append(line)
            
    with open(target_file, 'w') as f:
        f.writelines(new_lines)
    
    print(f"✅ SUCCESS: Fixed {fixed_count} lines in {target_file}")
else:
    print(f"❌ ERROR: File not found at {target_file}")