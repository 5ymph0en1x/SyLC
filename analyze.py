import ast

def analyze(file_path):
    with open(file_path, encoding='utf-8') as f:
        tree = ast.parse(f.read())
    
    print(f"--- {file_path} ---")
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            print(f"Class: {node.name}")
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    print(f"  def {item.name}()")
        elif isinstance(node, ast.FunctionDef):
            print(f"Function: {node.name}")

analyze('SyLC_3D_Player.py')
analyze('mvc_decoder.py')
