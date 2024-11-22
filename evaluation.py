import matplotlib.pyplot as plt
import pickle
import os

SCORES_PATH = "evaluation"

# Function to load scores from a file
def load_scores(path):
    # Check if the file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Score file not found at {path}")
    
    # Open and load scores from the file
    with open(path, 'rb') as scores_file:
        scores = pickle.load(scores_file)

    # Validate the scores data to ensure it's a list of numerical values
    if not all(isinstance(score, (int, float)) for score in scores):
        raise ValueError("The scores data contains invalid values. Scores must be numeric.")

    return scores

# Function to plot scores vs episodes
def scores_vs_episodes_plot(scores):
    episodes = [i for i in range(len(scores))]

    # Plot the graph
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, scores, label="Scores", color="b")

    plt.title("Scores vs Episodes")
    plt.xlabel("Episode #")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.show()

if '__main__' == __name__:
    try:
        # Load scores from the file and visualize
        scores = load_scores(SCORES_PATH)
        scores_vs_episodes_plot(scores)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
