"""Bounded causal flow-start windows; no labels, finished-flow values or model imports."""

import ipaddress
from collections import Counter, deque

WINDOWS = (1000, 10_000, 60_000)
CONTEXT_FEATURES = [
    *[f"src_flow_starts_{window // 1000}s" for window in WINDOWS],
    *[f"dst_flow_starts_{window // 1000}s" for window in WINDOWS],
    "src_distinct_destinations_60s",
    "src_distinct_dst_ports_60s",
    "dst_distinct_sources_60s",
]
CONTEXT_METADATA = [
    "context_init_sequence",
    "context_watermark_ms",
    "context_warmup_complete",
    "context_late_start",
    "context_peak_events",
]
SCHEMA = "nfstream-context-v1"


def endpoint(address):
    return ipaddress.ip_address(address).packed


def change(counter, key, delta):
    value = counter.get(key, 0) + delta
    if value < 0:
        raise ValueError("Context counter underflow.")
    if value:
        counter[key] = value
    else:
        counter.pop(key, None)


def nested_change(mapping, outer, inner, delta):
    if outer not in mapping:
        if delta < 0:
            raise ValueError("Context nested counter underflow.")
        mapping[outer] = Counter()
    change(mapping[outer], inner, delta)
    if not mapping[outer]:
        del mapping[outer]


class FlowStartContext:
    def __init__(self, max_events=100_000):
        if type(max_events) is not int or not 1 <= max_events <= 100_000:
            raise ValueError("Context event bound must be between one and 100k.")
        self.max_events = max_events
        self.queues = {window: deque() for window in WINDOWS}
        self.sources = {window: Counter() for window in WINDOWS}
        self.destinations = {window: Counter() for window in WINDOWS}
        self.source_peers, self.source_ports, self.destination_peers = {}, {}, {}
        self.pending = []
        self.watermark = self.first = None
        self.sequence = self.peak = 0
        self.failed = False

    def _commit_pending(self):
        for event in self.pending:
            _, source, destination, port = event
            for window in WINDOWS:
                self.queues[window].append(event)
                change(self.sources[window], source, 1)
                change(self.destinations[window], destination, 1)
            nested_change(self.source_peers, source, destination, 1)
            nested_change(self.source_ports, source, port, 1)
            nested_change(self.destination_peers, destination, source, 1)
        self.pending.clear()

    def _expire(self, timestamp):
        for window in WINDOWS:
            queue = self.queues[window]
            while queue and queue[0][0] < timestamp - window:
                _, source, destination, port = queue.popleft()
                change(self.sources[window], source, -1)
                change(self.destinations[window], destination, -1)
                if window == WINDOWS[-1]:
                    nested_change(self.source_peers, source, destination, -1)
                    nested_change(self.source_ports, source, port, -1)
                    nested_change(self.destination_peers, destination, source, -1)

    def observe(self, timestamp, source, destination, port):
        if self.failed:
            raise ValueError("Context state already failed; refuse a partial resumed capture.")
        if (
            type(timestamp) is not int
            or not 0 <= timestamp < 2**63
            or type(port) is not int
            or not 0 <= port <= 65535
            or not isinstance(source, bytes)
            or len(source) not in (4, 16)
            or not isinstance(destination, bytes)
            or len(destination) not in (4, 16)
        ):
            raise ValueError("Invalid context start time, packed endpoints or destination port.")
        late = self.watermark is not None and timestamp < self.watermark
        if self.watermark is None:
            self.first = self.watermark = timestamp
        elif timestamp > self.watermark:
            self._commit_pending()
            self.watermark = timestamp
            self._expire(timestamp)
        if len(self.queues[WINDOWS[-1]]) + len(self.pending) >= self.max_events:
            self.failed = True
            raise ValueError(
                "Context active/pending event bound reached; refuse incomplete history."
            )
        values = [self.sources[window].get(source, 0) for window in WINDOWS]
        values += [self.destinations[window].get(destination, 0) for window in WINDOWS]
        values += [
            len(self.source_peers.get(source, ())),
            len(self.source_ports.get(source, ())),
            len(self.destination_peers.get(destination, ())),
        ]
        self.sequence += 1
        self.pending.append((self.watermark, source, destination, port))
        self.peak = max(self.peak, len(self.queues[WINDOWS[-1]]) + len(self.pending))
        return {
            **dict(zip(CONTEXT_FEATURES, values, strict=True)),
            "context_init_sequence": self.sequence,
            "context_watermark_ms": self.watermark,
            "context_warmup_complete": int(self.watermark - self.first >= WINDOWS[-1]),
            "context_late_start": int(late),
            "context_peak_events": self.peak,
        }


def validate_context(row, max_events=100_000):
    for name in [*CONTEXT_FEATURES, *CONTEXT_METADATA]:
        if type(row[name]) is not int or not 0 <= row[name] < 2**63:
            raise ValueError("Context features and metadata must be exact nonnegative integers.")
    source = [row[name] for name in CONTEXT_FEATURES[:3]]
    destination = [row[name] for name in CONTEXT_FEATURES[3:6]]
    if (
        source != sorted(source)
        or destination != sorted(destination)
        or any(value > max_events for value in [*source, *destination])
        or row["src_distinct_destinations_60s"] > source[-1]
        or row["src_distinct_dst_ports_60s"] > source[-1]
        or row["dst_distinct_sources_60s"] > destination[-1]
        or row["context_peak_events"] > max_events
        or row["context_init_sequence"] == 0
        or row["context_warmup_complete"] not in (0, 1)
        or row["context_late_start"] not in (0, 1)
    ):
        raise ValueError("Context count, window, sequence or state invariants failed.")
