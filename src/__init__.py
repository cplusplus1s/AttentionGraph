import sys
import os

# 1. project_root/src
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. project_root
project_root = os.path.dirname(current_dir)
# 3. third_party
third_party_path = os.path.join(project_root, 'third_party')

# 4. Dynamically added to sys.path
if third_party_path not in sys.path:
    sys.path.append(third_party_path)
    # print(f"[System] Added {third_party_path} to Python path.")