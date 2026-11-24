"""Bounded, causal estimates of *served* output length, not future-work oracles.

Only completed output token counts enter history. Request max_new_tokens,
recorded outputs, and future tool durations are deliberately not consulted.
Estimates are shrunk toward the frozen prior when history is scarce. No local
clock influences a prediction, so identically completed TP histories agree.
"""
from collections import OrderedDict, deque
import math


class OutputHistory:
    def __init__(self, capacity=2048, window=64, minimum=4, quantile=0.75):
        if type(capacity) is not int or capacity < 1 or type(window) is not int or window < 1:
            raise ValueError("positive bounded history sizes required")
        if type(minimum) is not int or minimum < 1 or not 0 < quantile <= 1:
            raise ValueError("invalid history confidence configuration")
        self.capacity, self.window, self.minimum, self.quantile = capacity, window, minimum, quantile
        self.groups = OrderedDict()
        self.global_values = deque(maxlen=window)
        self.forecast_errors = deque(maxlen=max(32, 2 * window))

    def clear(self):
        self.groups.clear()
        self.global_values.clear()
        self.forecast_errors.clear()

    @staticmethod
    def keys(meta, input_tokens):
        lane = str(meta.get("lane", "unknown"))
        # These fields are available before generation. No request contents or
        # future length/finish metadata is used for grouping.
        scope = tuple(str(meta.get(k, "")) for k in ("family_id", "session_id", "scope_id"))
        bucket = max(0, int(input_tokens)).bit_length()
        keys = [("group", lane, bucket)]
        if scope[1] or scope[2]:
            keys.insert(0, ("scope", *scope, lane))
        return keys

    def observe(self, meta, input_tokens, output_tokens):
        if type(output_tokens) is not int or output_tokens <= 0:
            return False
        self.global_values.append(output_tokens)
        for key in self.keys(meta, input_tokens):
            values = self.groups.setdefault(key, deque(maxlen=self.window))
            values.append(output_tokens)
            self.groups.move_to_end(key)
            while len(self.groups) > self.capacity:
                self.groups.popitem(last=False)
        return True

    def predict(self, meta, input_tokens, prior):
        if not math.isfinite(prior) or prior < 0:
            raise ValueError("finite nonnegative prior required")
        choices = [(key[0], self.groups[key]) for key in self.keys(meta, input_tokens)
                   if key in self.groups]
        choices.append(("global", self.global_values))
        # Prefer a sufficiently supported narrow group; a small sample from a
        # narrow group must not override a much better supported broad group.
        supported = [(name, vals) for name, vals in choices if len(vals) >= self.minimum]
        name, values = supported[0] if supported else max(choices, key=lambda item: len(item[1]))
        if not values:
            return float(prior), 0, "frozen_prior"
        sample = sorted(values)[math.ceil(len(values) * self.quantile) - 1]
        weight = min(1., len(values) / self.minimum)
        return (1 - weight) * prior + weight * sample, len(values), name

    def record_forecast_error(self, online, prior, actual):
        """Score the forecasts saved at admission BEFORE adding this completion."""
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
               for v in (online, prior, actual)):
            raise ValueError("invalid causal forecast observation")
        self.forecast_errors.append((abs(online - actual), abs(prior - actual)))

    def guard(self, online, prior):
        """Global rolling backstop, not a calibrated confidence interval.

        32 previously completed forecasts and >=10% MAE improvement are required.
        Better length MAE is NOT a guarantee of better ranking or GPU throughput.
        """
        n = len(self.forecast_errors)
        if n < 32:
            return float(prior), "prior_insufficient_evidence"
        online_mae = sum(x[0] for x in self.forecast_errors) / n
        prior_mae = sum(x[1] for x in self.forecast_errors) / n
        if online_mae < 0.9 * prior_mae:
            return float(online), "online_past_error_improved"
        return float(prior), "prior_not_outperformed"

    def error_snapshot(self):
        n = len(self.forecast_errors)
        return {"completed_forecasts": n,
                "online_mae_tokens": sum(x[0] for x in self.forecast_errors) / n if n else None,
                "prior_mae_tokens": sum(x[1] for x in self.forecast_errors) / n if n else None,
                "semantics": "rolling admission forecasts; completion-causal; not ranking quality"}
