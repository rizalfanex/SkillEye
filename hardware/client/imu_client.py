"""
Computer-side client for the racket-mounted IMU (../firmware/firmware.ino).

The ESP32 is the host: it runs its own WiFi AP, and streaming starts the moment
this client opens a TCP connection to it. Rows arrive as CSV at 200 Hz:

    seq,t_us,ax,ay,az,gx,gy,gz

Standard library only -- no numpy, no install step. Python 3.8+.

Usage:

    python imu_client.py monitor                    # live check that it works
    python imu_client.py record --out take01.csv    # record until Ctrl-C
    python imu_client.py record --out take01.csv --seconds 10

The device address lives in config.local.json (gitignored; copy
config.example.json). --host / --port override it.

Note that the device clock (t_us) has no relationship to a camera's clock. Tap
the racket once at the start of every take -- that produces a spike in both the
accelerometer trace and the video, which is what the two get aligned on.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# ESP32 softAP always gateways at 192.168.4.1; config.local.json overrides.
DEFAULT_HOST = "192.168.4.1"
DEFAULT_PORT = 3333
CONFIG_FILE = Path(__file__).with_name("config.local.json")

IMU_COLUMNS = ("ax", "ay", "az", "gx", "gy", "gz")


def load_config():
    """Read the gitignored local config, if the user made one."""
    if not CONFIG_FILE.exists():
        return {}
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"warning: ignoring {CONFIG_FILE.name}: {exc}", file=sys.stderr)
        return {}


# --------------------------------------------------------------- parsing ----

def parse_metadata(line):
    """Pull `key=value` pairs out of one `# ...` header comment.

    Returns {} for prose comments, which the firmware also emits, so callers
    can feed it every comment line without pre-filtering.
    """
    body = line.lstrip("#").strip()
    out = {}
    for token in body.split():
        if token.count("=") != 1:
            continue
        key, value = token.split("=")
        if key and value:
            out[key] = value
    return out


@dataclass
class StreamStats:
    """Running health of one streaming session.

    `missing` is the one to watch: the firmware keeps incrementing seq even when
    it drops a sample it couldn't buffer, so a gap in seq is a truthful count of
    lost samples rather than an invisible hole in the timeline.
    """
    rows: int = 0
    missing: int = 0
    first_t_us: int | None = None
    last_t_us: int | None = None
    metadata: dict = field(default_factory=dict)
    _next_seq: int = 0

    @property
    def duration_s(self):
        if self.first_t_us is None or self.last_t_us is None:
            return 0.0
        return (self.last_t_us - self.first_t_us) / 1e6

    @property
    def measured_hz(self):
        """Rate implied by device timestamps, so it reflects the sampler's real
        cadence rather than how fast WiFi handed us the bytes."""
        span = self.duration_s
        if span <= 0 or self.rows < 2:
            return 0.0
        return (self.rows - 1) / span

    def note_row(self, seq, t_us):
        if seq > self._next_seq:
            self.missing += seq - self._next_seq
        self._next_seq = seq + 1
        self.rows += 1
        if self.first_t_us is None:
            self.first_t_us = t_us
        self.last_t_us = t_us


def iter_rows(lines, stats=None):
    """Parse the device's line stream into (seq, t_us, (ax..gz)) tuples.

    Takes any iterable of strings so the parser is testable without a socket.
    Comments and the column header are consumed silently; malformed rows are
    skipped rather than raising, because a half-written line at the moment of a
    disconnect shouldn't cost the take that came before it.
    """
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            if stats is not None:
                stats.metadata.update(parse_metadata(line))
            continue
        if line.startswith("seq,"):          # column header
            continue

        parts = line.split(",")
        if len(parts) != 2 + len(IMU_COLUMNS):
            continue
        try:
            seq = int(parts[0])
            t_us = int(parts[1])
            values = tuple(float(p) for p in parts[2:])
        except ValueError:
            continue

        if stats is not None:
            stats.note_row(seq, t_us)
        yield seq, t_us, values


# ---------------------------------------------------------------- socket ----

class IMUStream:
    """TCP connection to the device. Streaming begins on connect."""

    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT, timeout=10.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock = None

    def __enter__(self):
        self._sock = socket.create_connection((self.host, self.port), self.timeout)
        self._sock.settimeout(self.timeout)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return self

    def __exit__(self, *exc):
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        return False

    def lines(self):
        """Yield complete text lines, reassembling across packet boundaries."""
        buf = b""
        while True:
            try:
                chunk = self._sock.recv(4096)
            except socket.timeout:
                raise TimeoutError(
                    f"no data for {self.timeout:.0f}s -- is the device still powered "
                    f"and are you still joined to its WiFi network?"
                )
            if not chunk:
                break                                  # device closed the socket
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                yield line.decode("utf-8", errors="replace")
        if buf.strip():
            yield buf.decode("utf-8", errors="replace")


# ------------------------------------------------------------------ CLI ------

def _status_line(stats, wall_start):
    elapsed = time.monotonic() - wall_start
    drops = f", {stats.missing} dropped" if stats.missing else ""
    return (f"\r{stats.rows:7d} rows  {stats.measured_hz:6.1f} Hz  "
            f"{elapsed:6.1f}s{drops}   ")


def cmd_monitor(args):
    """Live sanity check: is it streaming, at what rate, and does the sensor
    actually respond when you move the racket?"""
    print(f"connecting to {args.host}:{args.port} ...", flush=True)
    stats = StreamStats()
    wall_start = time.monotonic()
    last_print = 0.0
    peak_g = 0.0

    with IMUStream(args.host, args.port, args.timeout) as stream:
        print("connected -- streaming. move the racket; Ctrl-C to stop.\n")
        for _, _, values in iter_rows(stream.lines(), stats):
            mag = math.sqrt(sum(v * v for v in values[:3]))
            peak_g = max(peak_g, mag)
            now = time.monotonic()
            if now - last_print >= 0.25:
                sys.stdout.write(_status_line(stats, wall_start)
                                 + f"|a|={mag:5.2f}g peak={peak_g:5.2f}g")
                sys.stdout.flush()
                last_print = now

    print()
    _print_summary(stats)


def cmd_record(args):
    """Record one take to CSV, byte-for-byte as the device sent it."""
    out_path = Path(args.out)
    if out_path.parent != Path(""):
        out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"connecting to {args.host}:{args.port} ...", flush=True)

    stats = StreamStats()
    wall_start = time.monotonic()
    last_print = 0.0

    with IMUStream(args.host, args.port, args.timeout) as stream, \
            open(out_path, "w", encoding="utf-8", newline="") as fh:
        print(f"connected -- recording to {out_path}")
        if args.seconds:
            print(f"stopping after {args.seconds:.0f}s (Ctrl-C to stop early)")
        else:
            print("Ctrl-C to stop")
        print("tap the racket now to mark the sync point.\n")

        try:
            for line in stream.lines():
                # Written verbatim, comments included -- the device's own
                # `# dropped=N` notes are part of the record's provenance.
                fh.write(line + "\n")
                for _ in iter_rows([line], stats):
                    pass

                now = time.monotonic()
                if now - last_print >= 0.25:
                    sys.stdout.write(_status_line(stats, wall_start))
                    sys.stdout.flush()
                    last_print = now
                if args.seconds and (now - wall_start) >= args.seconds:
                    break
        except KeyboardInterrupt:
            print("\nstopped")

    print()
    _print_summary(stats)
    print(f"wrote {out_path}")


def _print_summary(stats):
    if stats.metadata:
        meta = " ".join(f"{k}={v}" for k, v in sorted(stats.metadata.items()))
        print(f"device   : {meta}")
    print(f"rows     : {stats.rows}")
    print(f"duration : {stats.duration_s:.2f}s (device clock)")
    print(f"rate     : {stats.measured_hz:.1f} Hz")
    if stats.missing:
        pct = 100.0 * stats.missing / max(1, stats.missing + stats.rows)
        print(f"dropped  : {stats.missing} samples ({pct:.2f}%) -- gaps in seq")
    else:
        print("dropped  : none")


def main(argv=None):
    config = load_config()
    parser = argparse.ArgumentParser(
        description="Stream IMU data from the racket-mounted ESP32.")
    parser.add_argument("--host", default=config.get("host", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=config.get("port", DEFAULT_PORT))
    parser.add_argument("--timeout", type=float, default=10.0,
                        help="seconds of silence before giving up (default 10)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("monitor", help="live rate/peak readout, writes nothing")

    rec = sub.add_parser("record", help="record a take to CSV")
    rec.add_argument("--out", required=True, help="output CSV path")
    rec.add_argument("--seconds", type=float, default=None,
                     help="stop automatically after N seconds")

    args = parser.parse_args(argv)
    handler = {"monitor": cmd_monitor, "record": cmd_record}[args.command]

    try:
        handler(args)
    except (ConnectionRefusedError, OSError, TimeoutError) as exc:
        print(f"\nconnection failed: {exc}", file=sys.stderr)
        print("check: laptop joined the device's WiFi network? device LED "
              "blinking (waiting) rather than off?", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
