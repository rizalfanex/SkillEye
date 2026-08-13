"""Tests for imu_client.py. Standard library only:

    cd hardware/client && python -m unittest -v
"""

import socket
import threading
import unittest

from imu_client import IMUStream, StreamStats, iter_rows, parse_metadata

# Byte-for-byte what ../firmware/firmware.ino sends on connect.
DEVICE_HEADER = [
    "# skilleye imu stream v1",
    "# device=xiao_esp32c6 imu=mpu6050 addr=0x68",
    "# rate_hz=200 accel_range_g=16 gyro_range_dps=2000 dlpf_hz=94",
    "# units: accel in g, gyro in deg/s, t_us in microseconds since boot (monotonic)",
    "# no shared clock with the camera -- tap the racket once before each take",
    "seq,t_us,ax,ay,az,gx,gy,gz",
]


def make_rows(n, start_seq=0, period_us=5000):
    """n well-formed CSV rows at the firmware's 200 Hz cadence."""
    return [
        f"{start_seq + i},{(start_seq + i) * period_us},"
        f"{i * 0.01:.4f},{-1.0:.4f},{0.5:.4f},{1.0:.2f},{2.0:.2f},{3.0:.2f}"
        for i in range(n)
    ]


def serve_once(payload, chunk_size):
    """Stand in for the device: accept one connection, send `payload` in small
    chunks, hang up. Chunking is the point -- it forces the client to reassemble
    lines split across packets, which is what really happens at 200 Hz."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    def run():
        conn, _ = srv.accept()
        with conn:
            data = payload.encode("utf-8")
            for i in range(0, len(data), chunk_size):
                conn.sendall(data[i:i + chunk_size])
        srv.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return port, thread


class TestMetadata(unittest.TestCase):

    def test_extracts_pairs(self):
        meta = parse_metadata("# rate_hz=200 accel_range_g=16 gyro_range_dps=2000")
        self.assertEqual(
            meta, {"rate_hz": "200", "accel_range_g": "16", "gyro_range_dps": "2000"})

    def test_ignores_prose(self):
        # The firmware mixes prose comments in with the key=value ones; those
        # must not produce junk keys.
        self.assertEqual(parse_metadata("# tap the racket once before each take"), {})

    def test_collected_while_parsing(self):
        stats = StreamStats()
        list(iter_rows(DEVICE_HEADER + make_rows(3), stats))
        self.assertEqual(stats.metadata["rate_hz"], "200")
        # Prose lines (units, sync instructions) must not leak pseudo-keys into
        # what gets reported as the device's configuration.
        self.assertEqual(set(stats.metadata), {
            "device", "imu", "addr", "rate_hz",
            "accel_range_g", "gyro_range_dps", "dlpf_hz",
        })


class TestRowParsing(unittest.TestCase):

    def test_skips_comments_and_header(self):
        rows = list(iter_rows(DEVICE_HEADER + make_rows(4)))
        self.assertEqual(len(rows), 4)
        seq, t_us, values = rows[0]
        self.assertEqual((seq, t_us), (0, 0))
        self.assertEqual(len(values), 6)

    def test_preserves_channel_order(self):
        # ax..az then gx..gz, the MPU6050 register order.
        line = "7,35000,0.1000,0.2000,0.3000,4.00,5.00,6.00"
        (_, _, values), = list(iter_rows([line]))
        for got, want in zip(values, (0.1, 0.2, 0.3, 4.0, 5.0, 6.0)):
            self.assertAlmostEqual(got, want, places=4)

    def test_skips_malformed_without_raising(self):
        # A take shouldn't be lost to the half-written line a disconnect leaves.
        lines = (make_rows(2)
                 + ["3,15000,0.1,0.2", "4,20000,x,y,z,1,2,3", ""]
                 + make_rows(1, start_seq=5))
        self.assertEqual(len(list(iter_rows(lines))), 3)


class TestStats(unittest.TestCase):

    def test_counts_rows_and_rate(self):
        stats = StreamStats()
        list(iter_rows(make_rows(201), stats))       # 201 rows, 5000 us apart
        self.assertEqual(stats.rows, 201)
        self.assertAlmostEqual(stats.duration_s, 1.0, places=6)
        self.assertAlmostEqual(stats.measured_hz, 200.0, places=3)

    def test_detects_dropped_samples_from_seq_gap(self):
        # The device keeps incrementing seq through a drop, so the gap is the
        # truthful loss count -- this is the check that a take is intact.
        stats = StreamStats()
        list(iter_rows(make_rows(5) + make_rows(5, start_seq=12), stats))
        self.assertEqual(stats.rows, 10)
        self.assertEqual(stats.missing, 7)

    def test_clean_stream_reports_no_loss(self):
        stats = StreamStats()
        list(iter_rows(DEVICE_HEADER + make_rows(50), stats))
        self.assertEqual(stats.missing, 0)


class TestStreamOverSocket(unittest.TestCase):

    def test_end_to_end(self):
        payload = "\n".join(DEVICE_HEADER + make_rows(250)) + "\n"
        port, thread = serve_once(payload, chunk_size=7)

        stats = StreamStats()
        with IMUStream("127.0.0.1", port, timeout=5.0) as stream:
            rows = list(iter_rows(stream.lines(), stats))
        thread.join(timeout=5.0)

        self.assertEqual(len(rows), 250)
        self.assertEqual(stats.missing, 0)
        self.assertEqual(stats.metadata["rate_hz"], "200")
        self.assertEqual(rows[-1][0], 249)

    def test_survives_truncated_final_line(self):
        # Yanking power mid-row shouldn't cost the rows already received.
        payload = "\n".join(DEVICE_HEADER + make_rows(20)) + "\n5000,10000,0.1,0.2"
        port, thread = serve_once(payload, chunk_size=13)

        stats = StreamStats()
        with IMUStream("127.0.0.1", port, timeout=5.0) as stream:
            rows = list(iter_rows(stream.lines(), stats))
        thread.join(timeout=5.0)

        self.assertEqual(len(rows), 20)


if __name__ == "__main__":
    unittest.main()
