# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Particle Swarm Optimization (PSO)."""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from gradient_free_optimizers._array_backend import array, ndarray, zeros

from ._particle import Particle
from .base_population_optimizer import BasePopulationOptimizer

if TYPE_CHECKING:
    pass


class ParticleSwarmOptimizer(BasePopulationOptimizer):
    """Particle Swarm Optimization algorithm.

    Dimension Support:
        - Continuous: YES (velocity-based movement)
        - Categorical: YES (velocity interpreted as switch probability)
        - Discrete: YES (velocity-based with rounding)

    Simulates a swarm of particles moving through the search space. Each
    particle is influenced by its own best position (cognitive component)
    and the global best position found by the swarm (social component).

    The velocity update equation is:
        v = inertia * v
          + cognitive_weight * r1 * (personal_best - position)
          + social_weight * r2 * (global_best - position)
          + temp_weight * r3

    Template Method Pattern:
        This optimizer follows the Template Method Pattern by implementing
        _iterate_*_batch() methods instead of overriding iterate().
        The PSO velocity is computed once per iteration, then portions
        are extracted for each dimension type.

    Parameters
    ----------
    search_space : dict
        Dictionary mapping parameter names to search dimension definitions.
    initialize : dict, optional
        Strategy for generating initial positions.
    constraints : list, optional
        List of constraint functions.
    random_state : int, optional
        Seed for random number generation.
    rand_rest_p : float, default=0
        Probability of random restart.
    nth_process : int, optional
        Process index for parallel optimization.
    population : int, default=10
        Number of particles in the swarm.
    inertia : float, default=0.5
        Weight for maintaining current velocity direction (momentum).
    cognitive_weight : float, default=0.5
        Weight for attraction toward personal best position.
    social_weight : float, default=0.5
        Weight for attraction toward global best position.
    temp_weight : float, default=0.2
        Weight for small random velocity perturbations.
    """

    name = "Particle Swarm Optimization"
    _name_ = "particle_swarm_optimization"
    __name__ = "ParticleSwarmOptimizer"

    optimizer_type = "population"
    computationally_expensive = False

    def __init__(
        self,
        search_space: dict[str, Any],
        initialize: dict[str, int] | None = None,
        constraints: list[Callable[[dict[str, Any]], bool]] | None = None,
        random_state: int | None = None,
        rand_rest_p: float = 0,
        nth_process: int | None = None,
        boundary: str = "clip",
        population: int = 10,
        inertia: float = 0.5,
        cognitive_weight: float = 0.5,
        social_weight: float = 0.5,
        temp_weight: float = 0.2,
    ) -> None:
        super().__init__(
            search_space=search_space,
            initialize=initialize,
            constraints=constraints,
            random_state=random_state,
            rand_rest_p=rand_rest_p,
            nth_process=nth_process,
            boundary=boundary,
            population=population,
        )

        self.inertia = inertia
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.temp_weight = temp_weight

        # Create population of particles
        self.particles = self._create_population(Particle)
        self.optimizers = self.particles

        # Set particle parameters upfront on all particles
        n_dims = len(self.conv.search_space)
        for p in self.particles:
            p.inertia = self.inertia
            p.cognitive_weight = self.cognitive_weight
            p.social_weight = self.social_weight
            p.temp_weight = self.temp_weight
            p.rand_rest_p = self.rand_rest_p
            p.velo = zeros(n_dims)

        # Iteration state for template method coordination
        self._iteration_setup_done = False
        self._pso_new_pos = None
        self._pso_velocity_before_candidate = None

    def _collect_state(self) -> dict:
        """Expose the active particle's velocity magnitude for tracking.

        ``self.p_current`` is the particle whose velocity update produced the
        current candidate (set during iteration setup). The Euclidean norm of
        its velocity vector is a scalar swarm-convergence indicator: it decays
        toward zero as the swarm settles. Returns an empty dict before the
        first particle is active.
        """
        particle = self.p_current
        if particle is None or particle.velo is None:
            return {}
        velo_norm = sum(float(v) * float(v) for v in particle.velo) ** 0.5
        return {"velocity_norm": velo_norm}

    def _on_init_pos(self, position) -> None:
        """Initialize current particle with the given position.

        Sets up particle parameters (inertia, weights) and initializes
        velocity to zero for smooth startup. Assigns the position to
        the current particle.

        Args:
            position: The initialization position from CoreOptimizer.init_pos()
        """
        # Select particle via round-robin (use nth_init-1 since it was incremented)
        nth_pop = (self.nth_init - 1) % len(self.particles)
        self.p_current = self.particles[nth_pop]

        # Set particle parameters
        self.p_current.inertia = self.inertia
        self.p_current.cognitive_weight = self.cognitive_weight
        self.p_current.social_weight = self.social_weight
        self.p_current.temp_weight = self.temp_weight
        self.p_current.rand_rest_p = self.rand_rest_p

        # Initialize velocity to zero
        n_dims = self.conv.n_dimensions
        self.p_current.velo = zeros(n_dims)

        # Track position on current particle
        self.p_current._pos_new = position.copy()
        self.p_current._pos_current = position.copy()

    def _on_evaluate_init(self, score_new: float) -> None:
        """Evaluate during initialization phase.

        Delegates evaluation to the current particle for particle-level tracking.

        Args:
            score_new: Score of the most recently evaluated init position
        """
        # Track on current particle (this updates particle's best/current)
        self.p_current._score_new = score_new

        # Update particle's best if this is better
        if self.p_current._pos_best is None or score_new > self.p_current._score_best:
            self.p_current._pos_best = self.p_current._pos_new.copy()
            self.p_current._score_best = score_new

        # Update particle's current
        self.p_current._score_current = score_new

    def _sort_pop_personal_best_score(self) -> None:
        """Sort particles by their personal-best score, best first."""
        indexed = list(enumerate(self.particles))
        indexed.sort(
            key=lambda x: (
                x[1]._score_best if x[1]._pos_best is not None else float("-inf")
            ),
            reverse=True,
        )
        self.pop_sorted = [self.particles[i] for i, _ in indexed]

    def _setup_iteration(self) -> None:
        """Set up current iteration by selecting particle and computing velocity.

        Called lazily by the first _iterate_*_batch() method.
        Computes the full PSO position once, which is then extracted
        by the individual batch methods.
        """
        if self._iteration_setup_done:
            return

        # Select current particle (round-robin)
        self.p_current = self.particles[self.nth_trial % len(self.particles)]

        # Update global best reference for this particle
        self._sort_pop_personal_best_score()
        self.p_current.global_pos_best = self.pop_sorted[0]._pos_best

        # Compute full PSO position using velocity update
        self._pso_velocity_before_candidate = array(self.p_current.velo).copy()
        self._pso_new_pos = self._compute_pso_position()

        self._iteration_setup_done = True

    def _compute_pso_position(self) -> ndarray:
        """Compute new position using PSO velocity update equation.

        The velocity update equation is:
            v = inertia * v
              + cognitive_weight * r1 * (personal_best - position)
              + social_weight * r2 * (global_best - position)
              + temp_weight * r3

        Returns
        -------
        ndarray
            New position after velocity update.
        """
        # Random restart check
        if random.random() < self.rand_rest_p:
            return self.p_current.init.move_random_typed()

        # Guard against None positions during early iterations
        if (
            self.p_current._pos_current is None
            or self.p_current._pos_best is None
            or self.p_current.global_pos_best is None
        ):
            return self.p_current.init.move_random_typed()

        pos_current = array(self.p_current._pos_current)
        pos_best = array(self.p_current._pos_best)
        global_pos_best = array(self.p_current.global_pos_best)
        n_dims = len(pos_current)

        r1 = self._random_vector(n_dims)
        r2 = self._random_vector(n_dims)

        # Per-dimension random coefficients (Kennedy & Eberhart, 1995).
        # Using scalar r1/r2 couples all dimensions to the same random factor,
        # creating correlated diagonal movement and preventing independent
        # per-dimension exploration. Vectors ensure stochastic independence.
        n_dims = len(pos_current)
        r1 = array([random.random() for _ in range(n_dims)])
        r2 = array([random.random() for _ in range(n_dims)])

        # Inertia term: maintain current direction
        A = self.inertia * array(self.p_current.velo)

        # Cognitive term: attract toward personal best (per-dimension)
        B = self.cognitive_weight * r1 * (pos_best - pos_current)

        # Social term: attract toward global best (per-dimension)
        C = self.social_weight * r2 * (global_pos_best - pos_current)

        # Temperature term: add a bounded random vibration in position space
        D = self._compute_temperature_vibration(n_dims)

        # Update velocity
        self.p_current.velo = A + B + C + D

        # Compute new position (will be clipped by CoreOptimizer)
        return pos_current + self.p_current.velo

    def _compute_temperature_vibration(self, n_dims: int) -> ndarray:
        """Compute the stochastic velocity perturbation controlled by temp_weight."""
        if self.temp_weight == 0:
            return zeros(n_dims)

        return self.temp_weight * array(
            [random.uniform(-1.0, 1.0) for _ in range(n_dims)]
        )

    def _random_vector(self, n_dims: int) -> ndarray:
        """Sample per-dimension PSO random coefficients."""
        return array([random.random() for _ in range(n_dims)])

    def _reset_rejected_constraint_candidate(self) -> None:
        """Roll back non-evaluated PSO candidate state and clear the cache."""
        if self._pso_velocity_before_candidate is not None:
            self.p_current.velo = self._pso_velocity_before_candidate.copy()

        self._iteration_setup_done = False
        self._pso_new_pos = None

    def _on_constraint_retry(self, rejected_position) -> None:
        self._reset_rejected_constraint_candidate()

    def _on_constraint_retry_fallback(self) -> None:
        self._reset_rejected_constraint_candidate()

    def _on_constraint_retry_success(self, accepted_position) -> None:
        self._pso_velocity_before_candidate = None

    def _iterate_continuous_batch(self) -> ndarray:
        """Generate continuous values using PSO velocity update.

        Returns the continuous portion of the PSO-computed position.

        Returns
        -------
        ndarray
            New continuous values from PSO movement.
        """
        self._setup_iteration()
        return self._pso_new_pos[self._continuous_mask]

    def _iterate_categorical_batch(self) -> ndarray:
        """Generate categorical indices using PSO velocity as switch probability.

        For categorical dimensions, the velocity magnitude determines the
        probability of switching to a different category.

        Returns
        -------
        ndarray
            New category indices.
        """
        self._setup_iteration()

        # Get current categorical indices
        current = self.p_current._pos_current[self._categorical_mask]
        velocity = self.p_current.velo[self._categorical_mask]

        new_cats = []
        for i, (cur_idx, velo) in enumerate(zip(current, velocity)):
            # Get max valid index for this categorical dimension
            max_idx = self._categorical_sizes[i] - 1

            # Velocity magnitude determines switch probability
            # Normalize by category count to get reasonable probabilities
            switch_prob = min(1.0, abs(velo) / (max_idx + 1))

            if random.random() < switch_prob:
                # Switch to a random category
                new_cats.append(random.randint(0, max_idx))
            else:
                # Keep current category
                new_cats.append(int(cur_idx))

        return array(new_cats)

    def _iterate_discrete_batch(self) -> ndarray:
        """Generate discrete indices using PSO velocity update.

        Returns the discrete portion of the PSO-computed position.

        Returns
        -------
        ndarray
            New discrete indices from PSO movement.
        """
        self._setup_iteration()
        return self._pso_new_pos[self._discrete_mask]

    def _on_evaluate(self, score_new: float) -> None:
        """Evaluate current particle and update its personal/global best.

        Delegates to the particle's evaluate method which handles
        personal best tracking. Also resets iteration state for
        the next iteration.

        Parameters
        ----------
        score_new : float
            Score of the most recently evaluated position.
        """
        # Track position on particle (needed for personal best tracking)
        self.p_current._pos_new = self._pos_new

        # Delegate to current particle's evaluate
        self.p_current._evaluate(score_new)

        # Update global tracking
        self._update_best(self._pos_new, score_new)
        self._update_current(self._pos_new, score_new)

        # Reset iteration setup for next iteration
        self._iteration_setup_done = False
        self._pso_new_pos = None
        self._pso_velocity_before_candidate = None

    def _iterate_batch(self, n):
        """Generate n positions by cycling through particles."""
        self._sort_pop_personal_best_score()
        positions = []
        self._batch_particle_indices = []
        for i in range(n):
            idx = (self.nth_trial + i) % len(self.particles)
            self._batch_particle_indices.append(idx)
            self.p_current = self.particles[idx]
            self.p_current.global_pos_best = self.pop_sorted[0]._pos_best
            pos = self._compute_pso_position()
            positions.append(self._clip_position(pos))
        return positions

    def _evaluate_batch(self, positions, scores):
        """Process batch results, restoring the correct particle for each."""
        for i, (pos, score) in enumerate(zip(positions, scores)):
            self.p_current = self.particles[self._batch_particle_indices[i]]
            self._pos_new = pos
            self._evaluate(score)
