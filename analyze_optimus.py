#!/usr/bin/env python3
import os
import sys
import json
from pathlib import Path
from typing import Dict, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from utils import get_response


def extract_objective_from_code_output(code_output_content: str) -> Optional[float]:
    """
    Use GPT-4o-mini to extract objective value from code_output.txt.

    Args:
        code_output_content: The content of code_output.txt

    Returns:
        Extracted objective value as float, or None if extraction fails
    """
    prompt = f"""Extract the objective value from this optimization solver output.
The output contains the result of running an optimization problem.
Return ONLY the numeric objective value, nothing else. No text, no units, just the number.

Solver output:
{code_output_content}

Return only the number (e.g., 84.0 or 115000.0):"""

    try:
        response = get_response(prompt, model="gpt-4o-mini")
        return float(response.strip())
    except Exception as e:
        print(f"  Warning: Failed to extract objective from code_output.txt: {e}")
        return None


def analyze_optimus_data(base_path: str, model_type: str, model_name: str = None) -> Dict:
    """
    Analyze OptiMUS dataset and calculate accuracy for a specific model.

    Args:
        base_path: Path to the data directory containing problem folders
        model_type: Type of model run folder (e.g., 'google', 'qwen')
        model_name: Specific model to analyze within the run folder (e.g., 'gemini-2.5-flash')
                   If None, will analyze all models in the folder

    Returns:
        Dictionary containing analysis results
    """
    base_dir = Path(base_path)

    results = []
    infeasible_count = 0
    feasible_count = 0
    accurate_count = 0
    missing_output = 0
    missing_indices = []
    model_outputs = {}  # Track which models produced outputs

    # Get all problem folders (numeric only)
    problem_dirs = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda x: int(x.name)
    )
    print(f"\nAnalyzing {model_type.upper()} Model(s)")
    print(f"Found {len(problem_dirs)} problem folders\n")
    print("=" * 120)
    print(f"{'Problem':<15} {'Status':<15} {'Expected Obj':<20} {'Output Obj':<20} {'Match':<10}")
    print("=" * 120)

    for problem_dir in problem_dirs:
        problem_name = problem_dir.name
        solution_file = problem_dir / "solution.json"

        # Find the latest run folder (run_YYYYMMDD_{model_type})
        run_folders = sorted(
            [d for d in problem_dir.iterdir()
             if d.is_dir() and d.name.startswith("run_") and d.name.endswith(f"_{model_type}")],
            reverse=True
        )

        if not run_folders:
            missing_output += 1
            missing_indices.append(int(problem_name))
            print(f"{problem_name:<15} {'FEASIBLE':<15} {'N/A':<20} {'MISSING':<20} {'NO':<10}")
            results.append({
                'problem': problem_name,
                'status': 'feasible',
                'expected_obj': None,
                'output_obj': None,
                'match': False
            })
            continue

        latest_run_folder = run_folders[0]

        # Check if solution.json exists
        if not solution_file.exists():
            infeasible_count += 1
            print(f"{problem_name:<15} {'INFEASIBLE':<15} {'N/A':<20} {'N/A':<20} {'N/A':<10}")
            results.append({
                'problem': problem_name,
                'status': 'infeasible',
                'expected_obj': None,
                'output_obj': None,
                'match': False
            })
            continue

        feasible_count += 1

        # Load solution.json
        try:
            with open(solution_file, 'r') as f:
                solution_data = json.load(f)
            expected_obj = solution_data.get('objective')
        except Exception as e:
            print(f"{problem_name:<15} {'ERROR':<15} {'ERROR':<20} {'ERROR':<20} {'N/A':<10}")
            print(f"  Error reading solution.json: {e}")
            continue

        # Check output - o3 has output directly in run folder, others have subfolders
        output_obj = None
        found_output = False
        model_used = None

        # First, check if output_solution.txt exists directly in run folder (o3 case)
        direct_output_file = latest_run_folder / "output_solution.txt"
        if direct_output_file.exists():
            try:
                with open(direct_output_file, 'r') as f:
                    output_content = f.read().strip()
                output_obj = float(output_content.split()[-1])
                model_used = model_type  # Use model_type (e.g., 'o3') as the model name
                model_outputs[model_type] = model_outputs.get(model_type, 0) + 1
                found_output = True
            except Exception as e:
                pass

        # If not found directly, check in subfolders (Google, Qwen case)
        if not found_output:
            if model_name:
                models_to_check = [model_name]
            else:
                # Find all model folders in the run directory
                models_to_check = [d.name for d in latest_run_folder.iterdir() if d.is_dir()]

            for model in models_to_check:
                output_file = latest_run_folder / model / "output_solution.txt"
                if output_file.exists():
                    try:
                        with open(output_file, 'r') as f:
                            output_content = f.read().strip()
                        output_obj = float(output_content.split()[-1])
                        model_used = model
                        if model not in model_outputs:
                            model_outputs[model] = 0
                        model_outputs[model] += 1
                        found_output = True
                        break
                    except Exception as e:
                        continue

        # Fallback: If output_solution.txt not found, try extracting from code_output.txt
        if not found_output:
            code_output_file = latest_run_folder / "code_output.txt"
            if code_output_file.exists():
                try:
                    with open(code_output_file, 'r') as f:
                        code_output_content = f.read()
                    extracted_obj = extract_objective_from_code_output(code_output_content)
                    if extracted_obj is not None:
                        output_obj = extracted_obj
                        model_used = f"{model_type}_extracted"
                        model_outputs[model_used] = model_outputs.get(model_used, 0) + 1
                        found_output = True
                except Exception as e:
                    print(f"  Warning: Failed to read code_output.txt: {e}")

        if not found_output:
            missing_output += 1
            missing_indices.append(int(problem_name))
            print(f"{problem_name:<15} {'FEASIBLE':<15} {str(expected_obj):<20} {'MISSING':<20} {'NO':<10}")
            results.append({
                'problem': problem_name,
                'status': 'feasible',
                'expected_obj': expected_obj,
                'output_obj': None,
                'match': False
            })
            continue

        # Check if expected_obj is None
        if expected_obj is None:
            print(f"{problem_name:<15} {'FEASIBLE':<15} {'None':<20} {str(output_obj):<20} {'NO':<10}")
            results.append({
                'problem': problem_name,
                'status': 'feasible',
                'expected_obj': expected_obj,
                'output_obj': output_obj,
                'match': False
            })
            continue

        # Compare
        match = abs(float(expected_obj) - float(output_obj)) < 0.1
        if match:
            accurate_count += 1

        match_str = "YES" if match else "NO"
        print(f"{problem_name:<15} {'FEASIBLE':<15} {str(expected_obj):<20} {str(output_obj):<20} {match_str:<10}")

        results.append({
            'problem': problem_name,
            'status': 'feasible',
            'expected_obj': expected_obj,
            'output_obj': output_obj,
            'match': match
        })

    # Summary statistics
    print("=" * 120)
    print("\nSummary Statistics:")
    print(f"  Total problems:      {len(problem_dirs)}")
    print(f"  Infeasible:          {infeasible_count}")
    print(f"  Feasible:            {feasible_count}")
    print(f"  Missing output:      {missing_output}")
    print(f"  Accurate (matching): {accurate_count}")

    if feasible_count > 0:
        accuracy = (accurate_count / feasible_count) * 100
        print(f"  Accuracy:            {accuracy:.2f}% ({accurate_count}/{feasible_count})")

    if model_outputs:
        print(f"\n  Models found:")
        for model, count in model_outputs.items():
            print(f"    {model}: {count} outputs")

    # Show detailed results for mismatches
    mismatches = [r for r in results if r['status'] == 'feasible' and not r['match']]
    if mismatches:
        print(f"\nMismatches ({len(mismatches)}):")
        print("-" * 120)
        for r in mismatches:
            print(f"  {r['problem']}: Expected {r['expected_obj']}, Got {r['output_obj']}")

    # Show missing indices
    if missing_indices:
        print(f"\nMissing output indices ({len(missing_indices)}):")
        print(missing_indices)

    return {
        'model_type': model_type,
        'total_problems': len(problem_dirs),
        'infeasible': infeasible_count,
        'feasible': feasible_count,
        'accurate': accurate_count,
        'missing': missing_output,
        'accuracy': (accurate_count / feasible_count * 100) if feasible_count > 0 else 0,
        'results': results
    }

if __name__ == "__main__":
    # Configuration
    dataset = "ComplexLP"
    model = "o4-mini"  # e.g., "google/gemini-2.5-flash", "qwen/qwen3-30b-a3b", "o4-mini"

    base_path = f"/hpc/group/fanglab/xx102/OptiMUS/dataset/{dataset}"

    # Parse model string
    if "/" in model:
        model_type, model_name = model.split("/", 1)
    else:
        model_type = model
        model_name = None

    results = analyze_optimus_data(base_path, model_type=model_type, model_name=model_name)
