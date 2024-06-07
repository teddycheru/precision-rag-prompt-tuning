import os
import csv
import random
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Elo rating constants
K = 32  # K-factor determines the maximum possible adjustment per game

def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(rating_a, rating_b, result):
    expected_a = expected_score(rating_a, rating_b)
    new_rating_a = rating_a + K * (result - expected_a)
    return new_rating_a

# Monte Carlo simulation for matchups
def monte_carlo_simulation(prompts, num_simulations=1000):
    prompt_scores = {prompt: 0 for prompt in prompts}
    num_prompts = len(prompts)

    for _ in range(num_simulations):
        prompt_a, prompt_b = random.sample(prompts, 2)
        result = random.choice([0, 1])  # Simulate win/loss (0 for prompt_a loss, 1 for prompt_a win)
        prompt_scores[prompt_a] += result
        prompt_scores[prompt_b] += 1 - result

    # Normalize scores to probabilities
    total_simulations = num_simulations // 2  # Each prompt participates in approximately half of the total simulations
    for prompt in prompt_scores:
        prompt_scores[prompt] /= total_simulations

    return prompt_scores

def elo_ratings(prompts, initial_rating=1500, num_simulations=1000):
    ratings = {prompt: initial_rating for prompt in prompts}
    
    for _ in range(num_simulations):
        prompt_a, prompt_b = random.sample(prompts, 2)
        result = random.choice([0, 1])  # Simulate win/loss
        ratings[prompt_a] = update_elo(ratings[prompt_a], ratings[prompt_b], result)
        ratings[prompt_b] = update_elo(ratings[prompt_b], ratings[prompt_a], 1 - result)

    return ratings

def load_prompts_from_csv(csv_file):
    prompts = []
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            prompts.append(row[0])
    return prompts

def main():
    csv_file = os.path.join(os.path.dirname(__file__), "../../data/generated_prompts.csv")
    
    # Load prompts from CSV file
    loaded_prompts = load_prompts_from_csv(csv_file)
    
    # Evaluate prompts using Monte Carlo simulation
    monte_carlo_results = monte_carlo_simulation(loaded_prompts)
    
    # Output Monte Carlo evaluation results
    print("Monte Carlo Evaluation Results:")
    for prompt, score in monte_carlo_results.items():
        print(f"Prompt: '{prompt}' - Score: {score}")

    # Evaluate prompts using Elo rating system
    elo_results = elo_ratings(loaded_prompts)
    
    # Output Elo evaluation results
    print("\nElo Evaluation Results:")
    for prompt, elo_rating in elo_results.items():
        print(f"Prompt: '{prompt}' - Elo Rating: {elo_rating}")

if __name__ == "__main__":
    main()
