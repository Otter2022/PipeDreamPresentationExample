class Layer:
    """
    Represents a single layer in a deep neural network.

    Attributes:
      index (int): The index (position) of the layer in the network.
      compute_time (float): Time required for both forward and backward passes (Tₗ).
      activation_size (float): Size (e.g., in bytes) of the output activations produced by this layer (aₗ).
      weight_size (float): Size (e.g., in bytes) of the layer's weight parameters (wₗ).
    """
    def __init__(self, index, compute_time, activation_size, weight_size):
        self.index = index
        self.compute_time = compute_time
        self.activation_size = activation_size
        self.weight_size = weight_size


class Level:
    """
    Represents a hardware level in a hierarchical system (for example, a single GPU or a server).

    Attributes:
      level_id (int): Identifier for the level (e.g., 0 for a single device, 1 for a group/server level).
      workers (int): Number of compute devices available at this level.
      bandwidth (float or None): Communication bandwidth at this level (e.g., in bytes per time unit).
                                 For level 0 (a single device), this is None since no communication is needed.
    """
    def __init__(self, level_id, workers, bandwidth=None):
        self.level_id = level_id
        self.workers = workers
        self.bandwidth = bandwidth  # For level 0, bandwidth can be None


def A0(i, j):
    """
    Compute the baseline (device-level) compute time for layers i..j (inclusive).

    This is the sum of compute times (Tₗ) of layers with indices from i to j.

    Args:
      i (int): The starting layer index (0-based).
      j (int): The ending layer index (0-based).

    Returns:
      float: The total compute time (sum of Tₗ for layers i through j).
    """
    total = 0
    for k in range(i, j + 1):
        total += layers[k].compute_time
    return total


def dp(k, i, j, m):
    """
    Compute the optimal (minimal bottleneck) cost and partition for processing layers i..j 
    at hardware level k using m workers.

    This dynamic programming function recursively partitions the network into stages, 
    balancing computation (from lower levels) and communication costs. It returns both 
    the optimal cost (bottleneck time) and a partition (a list of tuples) indicating the stages 
    and transition (split) points.

    Args:
      k (int): The current hardware level. Level 0 is the base (a single device), while higher levels 
               represent groups of devices.
      i (int): The starting layer index (0-based) for the subproblem.
      j (int): The ending layer index (0-based) for the subproblem.
      m (int): The number of workers available at level k for processing layers i..j.

    Returns:
      tuple: A tuple (cost, partition) where:
          cost (float): The optimal (minimum bottleneck) cost for processing layers i..j.
          partition (list): A list of tuples describing the partition:
              - ("stage", i, j, m, k): Indicates a stage covering layers i..j at level k using m workers.
              - ("transition", s): Indicates a transition (split) at layer s.
    """
    global dp_cache
    key = (k, i, j, m)
    if key in dp_cache:
        return dp_cache[key]
    
    # Base case: at level 0, there is no communication, so m must be 1.
    if k == 0:
        cost = A0(i, j)
        partition = [("stage", i, j, 1, 0)]
        dp_cache[key] = (cost, partition)
        return cost, partition

    # If using only one worker at the current level, lift the solution from the lower level.
    if m == 1:
        result = dp(k - 1, i, j, levels[k - 1].workers)
        dp_cache[key] = result
        return result

    best = float('inf')
    best_partition = None
    # Try all possible split points s (from layer i to layer j-1).
    for s in range(i, j):
        # For each split, try all allocations: m_prime workers for the right sub-stage,
        # and (m - m_prime) workers for the left sub-stage.
        for m_prime in range(1, m):
            left_cost, left_part = dp(k, i, s, m - m_prime)
            # Transition cost: cost to communicate activations at layer s.
            trans_cost = (2 * layers[s].activation_size) / levels[k].bandwidth
            right_cost, right_part = T_func_partition(k, s + 1, j, m_prime)
            candidate = max(left_cost, trans_cost, right_cost)
            if candidate < best:
                best = candidate
                best_partition = left_part + [("transition", s)] + right_part
    result = (best, best_partition)
    dp_cache[key] = result
    return result


def T_func_partition(k, i, j, m):
    """
    Compute the effective stage time and partition for processing layers i..j 
    as a single stage at hardware level k using m workers.

    The effective time is defined as (1/m) times the maximum of:
      - The computation cost from the lower level: dp(k-1, i, j, levels[k-1].workers)
      - The communication cost: [2 * (m - 1) * total_weight] / levels[k].bandwidth,
        where total_weight is the sum of weight sizes (wₗ) for layers i..j.

    Args:
      k (int): The current hardware level.
      i (int): The starting layer index (0-based) for this stage.
      j (int): The ending layer index (0-based) for this stage.
      m (int): The number of workers allocated for this stage at level k.

    Returns:
      tuple: A tuple (cost, partition) where:
          cost (float): The effective stage time.
          partition (list): A list with a single tuple ("stage", i, j, m, k) representing this stage.
    """
    base_cost, base_part = dp(k - 1, i, j, levels[k - 1].workers)
    total_weight = 0
    for k_idx in range(i, j + 1):
        total_weight += layers[k_idx].weight_size
    comm_cost = (2 * (m - 1) * total_weight) / levels[k].bandwidth if m > 1 else base_cost
    cost = (1 / m) * max(base_cost, comm_cost)
    partition = [("stage", i, j, m, k)]
    return cost, partition


if __name__ == "__main__":
    layers = [
        Layer(index=1, compute_time=5, activation_size=1, weight_size=2),
        Layer(index=2, compute_time=3, activation_size=2, weight_size=2),
        Layer(index=3, compute_time=4, activation_size=1.5, weight_size=1),
        Layer(index=4, compute_time=6, activation_size=3, weight_size=3)
    ]
    n_layers = len(layers)


    level0 = Level(level_id=0, workers=1, bandwidth=2048)
    level1 = Level(level_id=1, workers=2, bandwidth=600)

    levels = [level0, level1]

    dp_cache = {}

    m_total = levels[1].workers  
    optimal_cost, partition = dp(1, 0, n_layers - 1, m_total)
    print("Optimal partition cost at level 1:", optimal_cost)
    print("Optimal stages (partitioning):")
    for p in partition:
        print(p)
