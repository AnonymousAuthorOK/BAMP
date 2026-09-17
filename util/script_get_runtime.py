import subprocess
import re
import os

json_files = [ 
    "ariane133.json",
    "ariane136.json",
    "black_parrot.json",
    "bp_be.json",
    "bp_fe.json",
    "bp_multi.json",
    "bp_quad.json",
    "swerv_wrapper.json"
]

log_file_path = "util/dreamplace_runtime.log"

with open(log_file_path, 'a') as log_file:
    for json_file in json_files:
        case_name = os.path.splitext(json_file)[0]
        command = f"python dreamplace/Placer.py test/or_cases/{json_file}"
        print(f"正在执行命令: {command}")
        
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        matched_lines = []
        
        for line in process.stdout:
            print(line, end='')  
            if re.search(r'\[INFO\s+\] DREAMPlace - .*takes .* seconds', line):
                matched_lines.append(line.rstrip())
        
        process.wait()
        if process.returncode == 0:
            print("命令执行成功")
        else:
            print(f"命令 {command} 执行失败，返回码: {process.returncode}")
        
        log_file.write(f"====={case_name}=====\n")
        for line in matched_lines:
            log_file.write(line + "\n")
        log_file.write("\n") 
