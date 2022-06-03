class Range:
    def __init__(self, min_vals, min_inclusive, max_vals, max_inclusive):
        assert min_vals.shape[0] == max_vals.shape[0]
        self.dim = min_vals.shape[0]
        self.min = min_vals
        self.min_inclusive = min_inclusive
        self.max = max_vals
        self.max_inclusive = max_inclusive

