"""Central configuration for the optimization experiments."""

RANDOM_STATE = 42

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

# Relevance-redundancy objective
REDUNDANCY_PENALTY = 0.10

# (candidate feature count, maximum selected features)
EXPERIMENTS = [
    (10, 5),
    (20, 10),
    (30, 15),
]

# ILP settings
ILP_TIME_LIMIT = 120  # seconds

# Genetic Algorithm settings
GA_POPULATION_SIZE = 60
GA_NUM_GENERATIONS = 100
GA_MUTATION_RATE = 0.05
GA_CROSSOVER_RATE = 0.80
GA_TOURNAMENT_SIZE = 3
GA_RUNS = 10

RESULTS_DIR = "results"