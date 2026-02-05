import os
import time
import json
import argparse
import glob
from concurrent.futures import ThreadPoolExecutor

from parameters import get_params
from constraint import get_constraints
from constraint_model import get_constraint_formulations
from target_code import get_codes
from generate_code import generate_code
from utils import load_state, save_state, Logger
from objective import get_objective
from objective_model import get_objective_formulation
from execute_code import execute_and_debug
from utils import create_state, get_labels
from rag.rag_utils import RAGMode
import litellm
litellm.suppress_debug_info = True

def process_single_dir(dir, devmode=1, rag_mode=None, error_correction=True, model="gpt-4o"):
    """Process a single directory through the optimization pipeline."""
    try:
        # Read the params state
        DEV_MODE = devmode
        RAG_MODE = rag_mode
        ERROR_CORRECTION = error_correction
        MODEL = model

        if DEV_MODE:
            run_dir = os.path.join(dir, f"run_{time.strftime('%Y%m%d')}_{MODEL}")
        else:
            # Get git hash
            git_hash = os.popen("git rev-parse HEAD").read().strip()
            run_dir = os.path.join(dir, f"run_{time.strftime('%Y%m%d')}_{MODEL}_{git_hash}_RAG")

        if not os.path.exists(run_dir):
            os.makedirs(run_dir)

        state = create_state(dir, run_dir, model)
        # labels = get_labels(dir)
        labels = None

        # Print the problem being solved
        print(f"\n{'='*80}")
        print(f"Processing problem: {os.path.basename(dir)}")
        print(f"Description: {state.get('description', 'N/A')}")
        print(f"{'='*80}\n")
    
        save_state(state, os.path.join(run_dir, "state_1_params.json"))

        # Save parameter values to data.json for code execution
        data = {}
        for param_name, param_data in state["parameters"].items():
            if "value" in param_data:
                data[param_name] = param_data["value"]

        with open(os.path.join(run_dir, "data.json"), "w") as f:
            json.dump(data, f, indent=4)

        logger = Logger(f"{run_dir}/log.txt")
        logger.reset()

        # # ###### Get objective
        state = load_state(os.path.join(run_dir, "state_1_params.json"))
        objective = get_objective(
            state["description"],
            state["parameters"],
            check=ERROR_CORRECTION,
            logger=logger,
            model=MODEL,
            rag_mode=RAG_MODE,
            labels=labels,
        )
        print(f"[{dir}] Objective: {objective}")
        state["objective"] = objective
        save_state(state, os.path.join(run_dir, "state_2_objective.json"))
        # #######
        # # # ####### Get constraints
        state = load_state(os.path.join(run_dir, "state_2_objective.json"))
        constraints = get_constraints(
            state["description"],
            state["parameters"],
            check=ERROR_CORRECTION,
            logger=logger,
            model=MODEL,
            rag_mode=RAG_MODE,
            labels=labels,
        )
        print(f"[{dir}] Constraints: {constraints}")
        state["constraints"] = constraints
        save_state(state, os.path.join(run_dir, "state_3_constraints.json"))
        # # # #######
        # ####### Get constraint formulations
        state = load_state(os.path.join(run_dir, "state_3_constraints.json"))
        constraints, variables = get_constraint_formulations(
            state["description"],
            state["parameters"],
            state["constraints"],
            check=ERROR_CORRECTION,
            logger=logger,
            model=MODEL,
            rag_mode=RAG_MODE,
            labels=labels,
        )
        state["constraints"] = constraints
        state["variables"] = variables
        save_state(state, os.path.join(run_dir, "state_4_constraints_modeled.json"))
        #######
        # ####### Get objective formulation
        state = load_state(os.path.join(run_dir, "state_4_constraints_modeled.json"))
        objective = get_objective_formulation(
            state["description"],
            state["parameters"],
            state["variables"],
            state["objective"],
            model=MODEL,
            check=ERROR_CORRECTION,
            rag_mode=RAG_MODE,
            labels=labels,
        )
        state["objective"] = objective
        print(f"[{dir}] DONE OBJECTIVE FORMULATION")
        save_state(state, os.path.join(run_dir, "state_5_objective_modeled.json"))
        # #######

        # # ####### Get codes
        state = load_state(os.path.join(run_dir, "state_5_objective_modeled.json"))
        constraints, objective = get_codes(
            state["description"],
            state["parameters"],
            state["variables"],
            state["constraints"],
            state["objective"],
            model=MODEL,
            check=ERROR_CORRECTION,
        )
        state["constraints"] = constraints
        state["objective"] = objective
        save_state(state, os.path.join(run_dir, "state_6_code.json"))
        # # #######

        ####### Run the code
        state = load_state(os.path.join(run_dir, "state_6_code.json"))
        generate_code(state, run_dir, problem_dir=dir)
        execute_and_debug(state, model=MODEL, dir=run_dir, logger=logger)
        #######

        return f"Successfully processed {dir}"
    except Exception as e:
        return f"Error processing {dir}: {str(e)}"


if __name__ == "__main__":
    # Hardcoded list of problem IDs with MISSING output to re-run
    MISSING_PROBLEM_IDS = [2]


    parser = argparse.ArgumentParser(description="Run the optimization problem")
    parser.add_argument("--dir", type=str, help="Directory of the problem")
    parser.add_argument("--all-dirs", action="store_true", help="Process all directories under /hpc/group/fanglab/xx102/OptiMUS/dataset/data/")
    parser.add_argument("--missing", action="store_true", help="Process hardcoded list of problems with MISSING output")
    parser.add_argument("--data-path", type=str, default="dataset/ComplexLP/", help="Base path for data directories")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of parallel workers for multiprocessing")
    parser.add_argument("--devmode", type=int, default=1)
    parser.add_argument("--rag-mode", type=RAGMode, choices=list(RAGMode), default=None, help="RAG mode")
    parser.add_argument("--model", type=str, default="o4-mini", help="Model to use (e.g., gpt-4o, o4-mini, gpt-5)")
    parser.add_argument("--error-correction", type=int, default=1, help="Enable error correction (1=True, 0=False)")
    args = parser.parse_args()

    # Read the params state
    DEV_MODE = args.devmode
    RAG_MODE = args.rag_mode
    ERROR_CORRECTION = bool(args.error_correction)
    MODEL = args.model

    if args.missing:
        # Process hardcoded list of problems with MISSING output
        dirs = [os.path.join(args.data_path, str(problem_id)) for problem_id in MISSING_PROBLEM_IDS]
        print(f"Processing {len(dirs)} problems with MISSING output: {MISSING_PROBLEM_IDS}")

        # Create process arguments
        process_args = [
            (dir, DEV_MODE, RAG_MODE, ERROR_CORRECTION, MODEL)
            for dir in dirs
        ]

        # Process in parallel
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            results = list(executor.map(lambda args: process_single_dir(*args), process_args))

        # Print results
        print("\n" + "="*80)
        print("PROCESSING COMPLETE")
        print("="*80)
        for result in results:
            print(result)
    elif args.all_dirs:
        # Get all directories under the data path
        dirs = sorted([d for d in glob.glob(os.path.join(args.data_path, "*")) if os.path.isdir(d)])
        print(f"Found {len(dirs)} directories to process")

        # Create process arguments
        process_args = [
            (dir, DEV_MODE, RAG_MODE, ERROR_CORRECTION, MODEL)
            for dir in dirs
        ]

        # Process in parallel
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            results = list(executor.map(lambda args: process_single_dir(*args), process_args))

        # Print results
        print("\n" + "="*80)
        print("PROCESSING COMPLETE")
        print("="*80)
        for result in results:
            print(result)
    else:
        if not args.dir:
            parser.error("Either --dir, --all-dirs, or --missing must be specified")
        process_single_dir(args.dir, DEV_MODE, RAG_MODE, ERROR_CORRECTION, MODEL)
