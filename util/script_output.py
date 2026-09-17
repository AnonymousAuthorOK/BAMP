import os
import re

# 定义缩放因子
SCALE_FACTOR = 2000.0

# 设计列表
DESIGNS = [
    "ariane133",
    # "ariane136",
    # "black_parrot", 
    # "bp_be",
    # "bp_fe",
    # "bp_multi", 
    # "bp_quad",
    # "swerv_wrapper"
]

# 根路径配置
RESULTS_DIR = "results"
BENCHMARKS_DIR = "benchmarks/or_cases"


def extract_component_locations(def_file, output_file, updated_output_file):
    try:
        # 读取 output 文件中的所有宏名称
        with open(output_file, 'r') as f:
            macros = [line.split('-macro_name')[-1].strip().split()[0] for line in f if '-macro_name' in line]
        macro_set = set(macros)  # 提升查找效率

        # 读取 def 文件，提取 components 部分
        component_locations = {}
        with open(def_file, 'r') as f:
            in_components_section = False
            component_data = ""
            for line in f:
                stripped = line.strip()
                if stripped.startswith('COMPONENTS'):
                    in_components_section = True
                elif stripped.startswith('END COMPONENTS'):
                    in_components_section = False

                if in_components_section:
                    component_data += stripped + " "  # 将多行拼接为一行

            # 正则匹配每个 component 定义
            components = re.findall(r"-\s*(\S+)\s+\S+.*?PLACED\s*\(\s*(\d+)\s+(\d+)\s*\)\s*[A-Z]+\s*;", component_data)
            for name, x, y in components:
                if name in macro_set:
                    # 转换坐标
                    new_x = float(x) / SCALE_FACTOR
                    new_y = float(y) / SCALE_FACTOR
                    component_locations[name] = (new_x, new_y)

        # 更新 output 文件
        with open(output_file, 'r') as f, open(updated_output_file, 'w') as out_f:
            for line in f:
                if '-macro_name' in line:
                    macro_name = line.split('-macro_name')[-1].strip().split()[0]
                    if macro_name in component_locations:
                        x, y = component_locations[macro_name]
                        new_line = f"place_macro -macro_name {macro_name} -location {{{x:.3f} {y:.3f}}} R0\n"
                        out_f.write(new_line)
                    else:
                        out_f.write(line)  # 保持原样
                else:
                    out_f.write(line)  # 保持原样
    except FileNotFoundError as e:
        print(f"Error: {e}")


# 批量处理所有设计
for design in DESIGNS:
    def_file = os.path.join(RESULTS_DIR, design, f"{design}.gp.def")
    output_file = os.path.join(BENCHMARKS_DIR, design, "mp_out")
    updated_output_file = os.path.join(BENCHMARKS_DIR, design, "updated_output")
    print(f"Processing {design}...")
    extract_component_locations(def_file, output_file, updated_output_file)
    print(f"Updated {updated_output_file}")

print("All designs processed.")
