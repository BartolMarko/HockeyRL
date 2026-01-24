import numpy as np

class SamplingStrategy:
    def __init__(self, cfg, num_arms, name="sampler"):
        self.cfg = cfg
        self.name = name
        self.num_arms = num_arms

    def sample(self, num_samples):
        raise NotImplementedError("Sample method must be implemented by subclasses.")

    def update_all_arms(self, new_weights):
        pass

    def update_arm(self, arm_idx, weight):
        assert 0 <= arm_idx < self.num_arms
        pass

    def remove_arm(self, arm_idx):
        assert 0 <= arm_idx < self.num_arms
        self.num_arms -= 1

    def add_arm(self, weight=None) -> int:
        self.num_arms += 1
        return 0

    def reset(self):
        pass

    def id(self):
        return self.name.lower().replace(" ", "_")

    def __len__(self):
        return self.num_arms

class UniformSampler(SamplingStrategy):
    def __init__(self, cfg, num_arms=0, name="uniform_sampler"):
        super().__init__(cfg, num_arms, name)

    def sample(self, num_samples):
        assert self.num_arms > 0, "No arms to sample from."
        num_samples = min(num_samples, self.num_arms)
        return np.random.choice(self.num_arms, size=num_samples)

    def get_weights(self):
        return [1.0 / self.num_arms] * self.num_arms if self.num_arms > 0 else []

class WeightedSampler(SamplingStrategy):
    def __init__(self, cfg, num_arms=0, name="weighted_sampler"):
        super().__init__(cfg, num_arms, name)
        self.weights = np.ones(self.num_arms)

    def add_arm(self, weight=1.0) -> int:
        if weight < 0:
            weight = self.weights.sum() if self.weights.size > 0 else 1.0
        self.weights = np.append(self.weights, weight)
        self.num_arms += 1
        return self.num_arms - 1

    def update_arm(self, arm_idx, weight):
        assert 0 <= arm_idx < self.num_arms
        if weight < 0:
            weight = self.weights.sum() if self.weights.size > 0 else 1.0
        self.weights[arm_idx] = weight

    def update_all_arms(self, new_weights):
        if isinstance(new_weights, list):
            new_weights = np.array(new_weights)
        assert len(new_weights) == self.num_arms
        self.weights = new_weights

    def sample(self, num_samples):
        assert self.num_arms > 0, "No arms to sample from."
        sampling_p = self.weights / self.weights.sum()
        return np.random.choice(self.num_arms, size=num_samples, p=sampling_p)

    def remove_arm(self, arm_idx):
        assert 0 <= arm_idx < self.num_arms
        self.weights = np.delete(self.weights, arm_idx)
        self.num_arms -= 1

    def get_weights(self):
        return self.weights.tolist()

class DeltaUniformSampler(SamplingStrategy):
    """
    Latest-arm gets extra probability mass (delta), while the rest share the remaining mass uniformly.
    """
    def __init__(self, cfg, num_arms=0, name="delta_uniform_sampler"):
        super().__init__(cfg, num_arms, name)
        self.delta = cfg.get("delta", 0.5) if cfg else 0.5

    def set_delta(self, new_delta):
        self.delta = new_delta

    def sample(self, num_samples):
        assert self.num_arms > 0, "No arms to sample from."

        base_prob = (1.0 - self.delta) / self.num_arms
        p = np.full(self.num_arms, base_prob)
        p[-1] += self.delta

        p = p / p.sum()
        return np.random.choice(self.num_arms, size=num_samples, p=p)

    def get_weights(self):
        if self.num_arms == 0:
            return []
        base_prob = (1.0 - self.delta) / self.num_arms
        p = np.full(self.num_arms, base_prob)
        p[-1] += self.delta
        return p.tolist()

class ModifiedDeltaUniformSampler(SamplingStrategy):
    """
    Sampling strategy where the Nth arm has delta prob mass, the N-1th arm has delta**2 prob mass, and so on.
    """
    def __init__(self, cfg, num_arms=0, name="delta_uniform_sampler"):
        super().__init__(cfg, num_arms, name)
        self.delta = cfg.get("delta", 0.5) if cfg else 0.5

    def set_delta(self, new_delta):
        self.delta = new_delta

    def sample(self, num_samples):
        assert self.num_arms > 0, "No arms to sample from."
        p = np.array([self.delta ** (self.num_arms - 1 - i) for i in range(self.num_arms)])
        p = p / p.sum()
        return np.random.choice(self.num_arms, size=num_samples, p=p)

    def get_weights(self):
        if self.num_arms == 0:
            return []
        p = np.array([self.delta ** (self.num_arms - 1 - i) for i in range(self.num_arms)])
        p = p / p.sum()
        return p.tolist()

class PrioritizedSelfPlaySampler(SamplingStrategy):
    def __init__(self, cfg, num_arms=0, name="pfsp_sampler"):
        super().__init__(cfg, num_arms, name)
        self.p = cfg.get("pfsp_p", 1.0) if cfg else 1.0
        self.weights = np.ones(num_arms) if num_arms > 0 else np.array([])

    def add_arm(self, weight=1.0) -> int:
        if weight < 0:
            if self.num_arms == 0:
                weight = 1.0
            else:
                weight = min(self.weights[-1] * 1.1, 1.0)
        self.weights = np.append(self.weights, weight)
        self.num_arms += 1
        return self.num_arms - 1

    def remove_arm(self, arm_idx):
        assert 0 <= arm_idx < self.num_arms
        self.weights = np.delete(self.weights, arm_idx)
        self.num_arms -= 1

    def update_arm(self, arm_idx, new_priority):
        assert 0 <= arm_idx < self.num_arms
        self.weights[arm_idx] = new_priority

    def update_all_arms(self, new_weights):
        if isinstance(new_weights, list):
            new_weights = np.array(new_weights)
        assert len(new_weights) == self.num_arms
        self.weights = new_weights

    def sample(self, num_samples):
        assert self.num_arms > 0, "No arms to sample from."

        # todo: what way to assign weights
        # alpha-star: f(w) = 1 - |0.5 - w|; w is win-rate
        alpha_star_weights = 1.0 - np.abs(0.5 - self.weights)
        sampling_p = alpha_star_weights ** self.p
        sampling_p /= sampling_p.sum()
        index = np.random.choice(self.num_arms, size=num_samples, p=sampling_p)
        return index

    def get_weights(self):
        return self.weights.tolist()

class TDWithRecencySampler(SamplingStrategy):
    """
    OK, Maybe this will work good:
    run one episode with each arm, get the abs-sum of TD-errors
    assign sampling prob proportional to TD-errors
    in addition, add recency bonus to the most recent arms
    priority = sum[|TD-error|] * (age)^-beta
    """
    # TODO: since i don't have the agents in memory, i will have to load all of them one by one
    #       and compute the TD-errors, which is very inefficient
    #       maybe i will work on it later
    def __init__(self, cfg, num_arms=0, name="td_recency_sampler"):
        super().__init__(cfg, num_arms, name)
        self.beta = cfg.get("recency_beta", 0.5) if cfg else 0.5
        raise NotImplementedError


def get_sampler_by_name(name, cfg=None, num_arms=0):
    name = name.lower()
    if name == "uniform":
        return UniformSampler(cfg, num_arms)
    elif name == "weighted":
        return WeightedSampler(cfg, num_arms)
    elif name == "delta_uniform":
        return DeltaUniformSampler(cfg, num_arms)
    elif name == "modified_delta_uniform":
        return ModifiedDeltaUniformSampler(cfg, num_arms)
    elif name == "pfsp":
        return PrioritizedSelfPlaySampler(cfg, num_arms)
    elif name == "td_recency":
        return TDWithRecencySampler(cfg, num_arms)
    else:
        raise ValueError(f"Unknown sampler name: {name}")

def get_sampler_using_config(cfg=None, num_arms=0):
    # TODO: need to better handle cfg here, unorganized way to set the params for sampler
    if cfg is None or "opponent_pool" not in cfg:
        name = "weighted"
    else:
        name = cfg["opponent_pool"].get("type", "weighted")
    return get_sampler_by_name(name, cfg, num_arms)

# ---- tests
def test_pfsp_sampler():
    print("Testing PrioritizedSelfPlaySampler...")
    cfg = {"pfsp_p": 0.8}
    sampler = PrioritizedSelfPlaySampler(cfg, num_arms=5)
    for i in range(5):
        sampler.update_arm(i, new_priority=i / 4.0)  # weights from 0.0 to 1.0
    counts = np.zeros(5)
    num_samples = 10000
    samples = sampler.sample(num_samples)
    for s in samples:
        counts[s] += 1
    # expected and actual sampling distribution
    print("Arm weights:", sampler.get_weights())
    print("Sampling distribution:", counts / num_samples)
    # Expected: arms with mid-range weights should be sampled more frequently
    assert counts[2] > counts[0] and counts[2] > counts[4], "Mid-range weight arm not sampled more frequently."

def test_delta_uniform_sampler():
    print("Testing DeltaUniformSampler...")
    cfg = {"delta": 0.6}
    sampler = DeltaUniformSampler(cfg, num_arms=4)
    counts = np.zeros(4)
    num_samples = 10000
    samples = sampler.sample(num_samples)
    for s in samples:
        counts[s] += 1
    print("Arm weights:", sampler.get_weights())
    print("Sampling distribution:", counts / num_samples)
    # Expected: last arm should have significantly higher sampling probability
    assert counts[3] > counts[0] and counts[3] > counts[1] and counts[3] > counts[2], "Last arm not sampled more frequently."

def test_modified_delta_uniform_sampler():
    print("Testing ModifiedDeltaUniformSampler...")
    cfg = {"delta": 0.45}
    sampler = ModifiedDeltaUniformSampler(cfg, num_arms=4)
    counts = np.zeros(4)
    num_samples = 10000
    samples = sampler.sample(num_samples)
    for s in samples:
        counts[s] += 1
    print("Arm weights:", sampler.get_weights())
    print("Sampling distribution:", counts / num_samples)
    # Expected: last arm should have highest sampling probability, followed by second last, etc.
    assert counts[3] > counts[2] > counts[1] > counts[0], "Sampling distribution does not match expected decay."

if __name__ == "__main__":
    test_pfsp_sampler()
    test_delta_uniform_sampler()
    test_modified_delta_uniform_sampler()
