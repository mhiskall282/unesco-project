"""Binary Whale Optimization Algorithm (BWOA) for Feature Selection.

This module implements the Binary Whale Optimization Algorithm (BWOA) used to
select an optimal subset of features for network intrusion detection.
The algorithm is based on the continuous Whale Optimization Algorithm by
Mirjalili and Lewis (2016), adapted for binary search spaces using a V-shaped
transfer function.
"""

from typing import Callable, Tuple, List
import numpy as np


class BinaryWhaleOptimizer:
    """Performs feature selection using the Binary Whale Optimization Algorithm."""

    def __init__(
        self,
        n_agents: int,
        n_features: int,
        max_iter: int,
        fitness_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], float],
        b: float = 1.0,
    ):
        """Initializes the Binary Whale Optimization Algorithm optimizer.

        Args:
            n_agents: Number of candidate solutions (whales) in the population.
            n_features: Dimension of the search space (total number of features).
            max_iter: Maximum number of search iterations.
            fitness_fn: A callable function to evaluate the fitness of a feature mask.
                Expected signature: fitness_fn(mask, X_train, y_train, X_val, y_val) -> float.
            b: Constant for defining the shape of the logarithmic spiral.
        """
        self.n_agents: int = n_agents
        self.n_features: int = n_features
        self.max_iter: int = max_iter
        self.fitness_fn: Callable = fitness_fn
        self.b: float = b

        # Initialize population with random binary masks (shape: n_agents, n_features)
        self.positions: np.ndarray = np.random.randint(0, 2, size=(self.n_agents, self.n_features))
        
        # Ensure that no whale has all features disabled (at least one feature selected)
        for i in range(self.n_agents):
            if np.sum(self.positions[i]) == 0:
                self.positions[i, np.random.randint(0, self.n_features)] = 1

    def _transfer_function(self, v: np.ndarray) -> np.ndarray:
        """Maps continuous values to probabilities using a V-shaped transfer function.

        The formula used is: T(v) = | v / sqrt(1 + v^2) |

        Args:
            v: Continuous step or velocity array.

        Returns:
            An array of probabilities mapped between 0 and 1.
        """
        return np.abs(v / np.sqrt(1.0 + np.square(v)))

    def _update_position(
        self,
        agent: np.ndarray,
        leader: np.ndarray,
        a: float,
        population: np.ndarray,
    ) -> np.ndarray:
        """Computes the continuous step update and applies the transfer function.

        This method selects between encircling, spiral update, and random search.

        Args:
            agent: The current search agent's position vector of shape (n_features,).
            leader: The best search agent's position vector of shape (n_features,).
            a: Parameter linearly decreasing from 2 to 0 over iterations.
            population: The entire population of search agents.

        Returns:
            The updated binary position vector.
        """
        p = np.random.rand()
        r1 = np.random.rand(self.n_features)
        r2 = np.random.rand(self.n_features)
        
        A = 2.0 * a * r1 - a
        C = 2.0 * r2
        l = np.random.uniform(-1.0, 1.0, size=self.n_features)
        
        # Compute continuous update step (velocity-like value)
        if p < 0.5:
            if np.all(np.abs(A) < 1.0):
                # Shrinking encircling mechanism
                D = np.abs(C * leader - agent)
                V = leader - A * D
            else:
                # Search for prey (exploration) using a random whale
                random_index = np.random.randint(0, self.n_agents)
                random_agent = population[random_index]
                D = np.abs(C * random_agent - agent)
                V = random_agent - A * D
        else:
            # Spiral bubble-net attack
            D_prime = np.abs(leader - agent)
            V = D_prime * np.exp(self.b * l) * np.cos(2.0 * np.pi * l) + leader

        # Apply V-shaped transfer function to convert continuous step to probabilities
        prob = self._transfer_function(V)
        
        # Determine whether to flip bits based on probability threshold
        r3 = np.random.rand(self.n_features)
        new_agent = np.where(r3 < prob, 1 - agent, agent)
        
        # Ensure at least one feature remains selected
        if np.sum(new_agent) == 0:
            new_agent[np.random.randint(0, self.n_features)] = 1
            
        return new_agent

    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[np.ndarray, List[float]]:
        """Orchestrates the BWOA optimization search iterations.

        Args:
            X_train: Training features array.
            y_train: Training labels array.
            X_val: Validation features array.
            y_val: Validation labels array.

        Returns:
            A tuple of (best_feature_mask, best_fitness_history) where:
                best_feature_mask: Binary array indicating chosen features.
                best_fitness_history: Record of best fitness values per iteration.
        """
        best_fitness_history: List[float] = []
        best_fitness = float("inf")
        best_agent = np.copy(self.positions[0])

        # Evaluate initial population fitness
        for i in range(self.n_agents):
            fitness = self.fitness_fn(self.positions[i], X_train, y_train, X_val, y_val)
            if fitness < best_fitness:
                best_fitness = fitness
                best_agent = np.copy(self.positions[i])

        # Iterative search loop
        for iteration in range(self.max_iter):
            # Parameter a decreases linearly from 2 to 0
            a = 2.0 - 2.0 * (iteration / self.max_iter)
            
            new_positions = np.zeros_like(self.positions)
            
            for i in range(self.n_agents):
                new_positions[i] = self._update_position(
                    self.positions[i],
                    best_agent,
                    a,
                    self.positions
                )
                
            # Evaluate new positions
            for i in range(self.n_agents):
                fitness = self.fitness_fn(new_positions[i], X_train, y_train, X_val, y_val)
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_agent = np.copy(new_positions[i])
                    
            self.positions = new_positions
            best_fitness_history.append(best_fitness)
            print(f"Iteration {iteration + 1}/{self.max_iter} - Best Fitness: {best_fitness:.5f}")

        return best_agent, best_fitness_history
