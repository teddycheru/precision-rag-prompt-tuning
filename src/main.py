import subprocess

def main():
    iteration = 1
    max_iterations = 5
    
    while iteration <= max_iterations:
        print(f"--- Iteration {iteration} ---")
        
        # Generate prompts
        subprocess.run(["python3", "src/prompt_generation/prompt_generator.py"])
        
        # Evaluate prompts
        subprocess.run(["python3", "src/evaluation/evaluator.py"])
   
        iteration += 1
    
    print("Pipeline completed.")

if __name__ == "__main__":
    main()
