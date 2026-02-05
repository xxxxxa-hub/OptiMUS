import os
import numpy as np
import json


def get_var_code(symbol, shape, type, definition, solver="gurobipy"):

    if solver == "gurobipy":
        if shape == []:
            return (
                f'{symbol} = model.addVar(vtype=GRB.{type.upper()}, name="{symbol}")\n'
            )
        else:
            return (
                f"{symbol} = model.addVars("
                + ", ".join([str(i) for i in shape])
                + f', vtype=GRB.{type.upper()}, name="{symbol}")\n'
            )
    else:
        raise NotImplementedError(f"Solver {solver} is not implemented")


def get_param_code(symbol, shape, definition):
    return f'{symbol} = data["{symbol}"] # shape: {shape}, definition: {definition}\n'


def generate_code(state, dir, problem_dir=None):
    code = []
    code.append(
        f"""
import os
import numpy as np
import json 
from gurobipy import Model, GRB, quicksum


model = Model("OptimizationProblem")

with open("data.json", "r") as f:
    data = json.load(f)

"""
    )
    problem_info = None
    if problem_dir and os.path.exists(os.path.join(problem_dir, "problem_info.json")):
        with open(os.path.join(problem_dir, "problem_info.json"), "r") as f:
            problem_info = json.load(f)

    code.append("\n\n### Define the parameters\n")
    for symbol, v in state["parameters"].items():
        print(v)
        # Use ground truth from problem_info.json if available, otherwise use state
        if problem_info and "parameters" in problem_info and symbol in problem_info["parameters"]:
            shape = problem_info["parameters"][symbol]["shape"]
            definition = problem_info["parameters"][symbol]["description"]
        else:
            shape = v["shape"]
            definition = v["definition"]
        code.append(get_param_code(symbol, shape, definition))

    code.append("\n\n### Define the variables\n")
    for symbol, v in state["variables"].items():
        code.append(
            get_var_code(
                symbol,
                v["shape"],
                v["type"],
                v["definition"],
                solver="gurobipy",
            )
        )

    code.append("\n\n### Define the constraints\n")
    for c in state["constraints"]:
        code.append(c["code"])

    code.append("\n\n### Define the objective\n")
    code.append(state["objective"]["code"])

    code.append("\n\n### Optimize the model\n")
    code.append("model.optimize()\n")

    code.append("\n\n### Output optimal objective value\n")
    code.append(f'print("Optimal Objective Value: ", model.objVal)\n')

    # code to save the optimal value if it exists
    code.append(
        """
if model.status == GRB.OPTIMAL:
    with open("output_solution.txt", "w") as f:
        f.write(str(model.objVal))
    print("Optimal Objective Value: ", model.objVal)
else:
    with open("output_solution.txt", "w") as f:
        f.write(model.status)
"""
    )

    code_str = "\n".join(code)

    with open(os.path.join(dir, "code.py"), "w") as f:
        f.write(code_str)
