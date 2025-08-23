import os
import shutil

def filter(input_folder, output_folder):
    files = os.listdir(input_folder)
    files_with_labels = {os.path.splitext(f)[0] for f in files if f.endswith('.txt')}

    for f in files:
        name, ext = os.path.splitext(f)
        if name in files_with_labels and (ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp'] or ext == '.txt'):
            src_path = os.path.join(input_folder, f)
            dst_path = os.path.join(output_folder, f)
            shutil.copy2(src_path, dst_path)

    print(f"Saved to: {output_folder}")

input_dir = 'unfiltered'    
output_dir = 'filtered' 
filter(input_dir, output_dir)
