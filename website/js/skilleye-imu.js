/*!
 * skilleye-imu.js -- IMU analysis engine for the SkillEye website.
 *
 * Pure computation, no DOM: parses the racket IMU stream written by
 * hardware/firmware/firmware.ino (via hardware/client/imu_client.py or
 * sync_recorder.py), segments it into taps and swings, and derives the
 * physics-based per-swing metrics the camera cannot see.
 *
 * Everything here is closed-form physics + descriptive statistics. No model
 * is trained or loaded; nothing here needs the Python backend.
 *
 * Wire format consumed (firmware.ino "skilleye imu stream v1"):
 *
 *     # skilleye imu stream v1
 *     # rate_hz=200 accel_range_g=16 gyro_range_dps=2000 dlpf_hz=94
 *     seq,t_us,ax,ay,az,gx,gy,gz
 *     0,640643845,0.5317,-0.5210,0.7827,58.23,-45.98,1.40
 *
 * accel in g, gyro in deg/s, t_us monotonic microseconds since device boot.
 *
 * Loads as a browser global (window.SkillEyeIMU) or a CommonJS module.
 */
(function (root, factory) {
    if (typeof module === 'object' && module.exports) module.exports = factory();
    else root.SkillEyeIMU = factory();
}(typeof self !== 'undefined' ? self : this, function () {
    'use strict';

    var G = 9.80665;                 // m/s^2 per g
    var DEG = Math.PI / 180;

    // ---------------------------------------------------------------- utils --

    function hann(width) {
        var w = width | 1, k = new Float64Array(w), s = 0, i;
        for (i = 0; i < w; i++) { k[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (w - 1)); s += k[i]; }
        for (i = 0; i < w; i++) k[i] /= s;
        return k;
    }

    /** Zero-phase-ish smoothing with a Hann kernel, edges clamped. */
    function smooth(x, width) {
        if (width <= 1) return Float64Array.from(x);
        var k = hann(width), h = (k.length - 1) >> 1, n = x.length;
        var out = new Float64Array(n);
        for (var i = 0; i < n; i++) {
            var acc = 0, wsum = 0;
            for (var j = 0; j < k.length; j++) {
                var idx = i + j - h;
                if (idx < 0 || idx >= n) continue;
                acc += x[idx] * k[j]; wsum += k[j];
            }
            out[i] = wsum > 0 ? acc / wsum : x[i];
        }
        return out;
    }

    function median(arr) {
        if (!arr.length) return NaN;
        var a = Array.prototype.slice.call(arr).sort(function (p, q) { return p - q; });
        var m = a.length >> 1;
        return a.length % 2 ? a[m] : 0.5 * (a[m - 1] + a[m]);
    }

    function percentile(arr, p) {
        if (!arr.length) return NaN;
        var a = Array.prototype.slice.call(arr).sort(function (x, y) { return x - y; });
        var i = (a.length - 1) * p / 100, lo = Math.floor(i), hi = Math.ceil(i);
        return lo === hi ? a[lo] : a[lo] + (a[hi] - a[lo]) * (i - lo);
    }

    function mean(arr) {
        if (!arr.length) return NaN;
        var s = 0; for (var i = 0; i < arr.length; i++) s += arr[i];
        return s / arr.length;
    }

    function stdev(arr) {
        var n = arr.length; if (n < 2) return NaN;
        var m = mean(arr), s = 0;
        for (var i = 0; i < n; i++) s += (arr[i] - m) * (arr[i] - m);
        return Math.sqrt(s / (n - 1));   // sample std -- these are samples of a session
    }

    /** Coefficient of variation (%) -- unit-free spread, so metrics in deg/s,
     *  degrees and milliseconds can be compared on one scale. */
    function cv(arr) {
        var m = mean(arr);
        if (!isFinite(m) || Math.abs(m) < 1e-9) return NaN;
        return 100 * stdev(arr) / Math.abs(m);
    }

    // -------------------------------------------------------------- parsing --

    /**
     * Parse a raw IMU CSV (the exact text written by imu_client.py / sync_recorder.py).
     * Returns typed arrays plus stream health, or throws on an unusable file.
     */
    function parseImuCsv(text) {
        var lines = String(text).split(/\r?\n/);
        var meta = {}, seq = [], t = [], ax = [], ay = [], az = [], gx = [], gy = [], gz = [];
        var headerSeen = false, badRows = 0;

        for (var li = 0; li < lines.length; li++) {
            var line = lines[li].trim();
            if (!line) continue;
            if (line.charAt(0) === '#') {
                // "# rate_hz=200 accel_range_g=16 ..." -> meta.rate_hz = 200
                var re = /([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)/g, m;
                while ((m = re.exec(line)) !== null) {
                    var v = Number(m[2]);
                    meta[m[1]] = isFinite(v) && m[2] !== '' ? v : m[2];
                }
                continue;
            }
            if (!headerSeen && /^seq\s*,/i.test(line)) { headerSeen = true; continue; }
            var p = line.split(',');
            if (p.length < 8) { badRows++; continue; }
            var row = [+p[0], +p[1], +p[2], +p[3], +p[4], +p[5], +p[6], +p[7]];
            var ok = true;
            for (var c = 0; c < 8; c++) if (!isFinite(row[c])) { ok = false; break; }
            if (!ok) { badRows++; continue; }
            seq.push(row[0]); t.push(row[1]);
            ax.push(row[2]); ay.push(row[3]); az.push(row[4]);
            gx.push(row[5]); gy.push(row[6]); gz.push(row[7]);
        }

        if (t.length < 50) {
            throw new Error('IMU CSV has only ' + t.length + ' usable rows -- expected hundreds. ' +
                'Is this a skilleye imu stream file (seq,t_us,ax,ay,az,gx,gy,gz)?');
        }

        var n = t.length, i;
        var t0 = t[0];
        var ts = new Float64Array(n);
        for (i = 0; i < n; i++) ts[i] = (t[i] - t0) / 1e6;   // seconds since first sample

        var accMag = new Float64Array(n), gyrMag = new Float64Array(n);
        for (i = 0; i < n; i++) {
            accMag[i] = Math.sqrt(ax[i] * ax[i] + ay[i] * ay[i] + az[i] * az[i]);
            gyrMag[i] = Math.sqrt(gx[i] * gx[i] + gy[i] * gy[i] + gz[i] * gz[i]);
        }

        var duration = ts[n - 1];
        var expected = seq[n - 1] - seq[0] + 1;
        var dropped = Math.max(0, expected - n);
        var fs = duration > 0 ? (n - 1) / duration : (meta.rate_hz || 200);

        // Saturation: the configured full-scale range, minus a small margin.
        var aRange = meta.accel_range_g || 16, gRange = meta.gyro_range_dps || 2000;
        var satA = 0, satG = 0;
        for (i = 0; i < n; i++) {
            if (Math.abs(ax[i]) > aRange * 0.995 || Math.abs(ay[i]) > aRange * 0.995 ||
                Math.abs(az[i]) > aRange * 0.995) satA++;
            if (Math.abs(gx[i]) > gRange * 0.995 || Math.abs(gy[i]) > gRange * 0.995 ||
                Math.abs(gz[i]) > gRange * 0.995) satG++;
        }

        return {
            meta: meta, n: n, fs: fs, duration: duration,
            t: ts, seq: Float64Array.from(seq),
            ax: Float64Array.from(ax), ay: Float64Array.from(ay), az: Float64Array.from(az),
            gx: Float64Array.from(gx), gy: Float64Array.from(gy), gz: Float64Array.from(gz),
            accMag: accMag, gyrMag: gyrMag,
            health: {
                rows: n, dropped: dropped, badRows: badRows,
                dropRate: expected > 0 ? dropped / expected : 0,
                effectiveHz: fs, nominalHz: meta.rate_hz || null,
                saturatedAccel: satA, saturatedGyro: satG
            }
        };
    }

    // ---------------------------------------------------- event segmentation --

    var DEFAULTS = {
        // A tap is a sharp impact with no rotation -- the sync marker the
        // firmware header asks the user to produce before/after each take.
        tapMinG: 2.5,
        tapMaxGyroDps: 250,        // smoothed |gyro| in the tap's neighbourhood
        tapContextS: 0.20,
        // A swing is a sustained rotation burst.
        swingSmoothS: 0.20,        // smoothing window for the |gyro| envelope
        swingMinPeakDps: 150,      // absolute floor; lowered so low-energy volleys fire
        swingRelHeight: 0.45,      // ... and at least this fraction of the take's p99
        swingMinPeakG: 1.2,        // a real swing also throws real acceleration --
                                    // lowered so compact volleys (low g) are kept
        swingMinSepS: 0.4,        // separate reps done close together (fixes swing count)
        swingEdgeFrac: 0.25,       // burst edges where the envelope crosses this
                                   // fraction of its own peak
        swingMaxEdgeS: 0.90,       // ... but never run the edge search further than this
        swingTapGuardS: 0.30       // a burst peaking this close to a tap is the tap,
                                   // not a swing
    };

    function findPeaks(x, height, minSep) {
        var out = [], n = x.length;
        for (var i = 1; i < n - 1; i++) {
            if (x[i] >= height && x[i] >= x[i - 1] && x[i] > x[i + 1]) out.push(i);
        }
        // greedy non-maximum suppression by descending amplitude
        out.sort(function (a, b) { return x[b] - x[a]; });
        var kept = [];
        for (var k = 0; k < out.length; k++) {
            var ok = true;
            for (var j = 0; j < kept.length; j++) {
                if (Math.abs(out[k] - kept[j]) < minSep) { ok = false; break; }
            }
            if (ok) kept.push(out[k]);
        }
        kept.sort(function (a, b) { return a - b; });
        return kept;
    }

    /**
     * Taps: narrow |accel| spikes with (almost) no angular rate around them.
     * Used to mark the start/end of a take, and to reject impacts that are not
     * swings. Not used as the camera sync anchor -- see skilleye-sync.js for why.
     */
    function detectTaps(imu, opts) {
        var o = Object.assign({}, DEFAULTS, opts || {});
        var ctx = Math.max(1, Math.round(o.tapContextS * imu.fs));
        var gs = smooth(imu.gyrMag, Math.max(3, Math.round(0.10 * imu.fs)));
        var cand = findPeaks(imu.accMag, o.tapMinG, Math.max(2, Math.round(0.05 * imu.fs)));
        var taps = [];
        for (var c = 0; c < cand.length; c++) {
            var p = cand[c], lo = Math.max(0, p - ctx), hi = Math.min(imu.n, p + ctx);
            var gmax = 0;
            for (var i = lo; i < hi; i++) if (gs[i] > gmax) gmax = gs[i];
            if (gmax < o.tapMaxGyroDps) {
                taps.push({ index: p, t: imu.t[p], accG: imu.accMag[p], gyroContextDps: gmax });
            }
        }
        return taps;
    }

    /**
     * Swings: rotation bursts in the |gyro| envelope. Each swing carries the
     * indices of its own start / peak-rotation / end so every later metric is
     * computed on that swing's own window only.
     */
    function detectSwings(imu, opts, taps) {
        var o = Object.assign({}, DEFAULTS, opts || {});
        var env = smooth(imu.gyrMag, Math.max(3, Math.round(o.swingSmoothS * imu.fs)));
        var p99 = percentile(env, 99);
        var height = Math.max(o.swingMinPeakDps, p99 * o.swingRelHeight);
        var peaks = findPeaks(env, height, Math.round(o.swingMinSepS * imu.fs));

        // A tap is a bare impact: it can push the envelope up without being a
        // swing. Drop any burst whose peak sits on top of one.
        var tapList = taps || detectTaps(imu, o);
        peaks = peaks.filter(function (p) {
            for (var i = 0; i < tapList.length; i++) {
                if (Math.abs(imu.t[p] - tapList[i].t) < o.swingTapGuardS) return false;
            }
            return true;
        });

        var maxEdge = Math.max(2, Math.round(o.swingMaxEdgeS * imu.fs));
        var swings = [];
        for (var k = 0; k < peaks.length; k++) {
            var p = peaks[k];
            var edge = env[p] * o.swingEdgeFrac;
            // Never cross into a neighbouring burst, and never run away when two
            // bursts merge -- clamp to the midpoint and to swingMaxEdgeS.
            var loBound = Math.max(0, p - maxEdge);
            var hiBound = Math.min(imu.n - 1, p + maxEdge);
            if (k > 0) loBound = Math.max(loBound, (peaks[k - 1] + p) >> 1);
            if (k < peaks.length - 1) hiBound = Math.min(hiBound, (p + peaks[k + 1]) >> 1);

            var s = p, e = p;
            while (s > loBound && env[s] > edge) s--;
            while (e < hiBound && env[e] > edge) e++;

            // impact index = strongest |accel| inside the burst
            var iAcc = s, aMax = -1;
            for (var i = s; i <= e; i++) if (imu.accMag[i] > aMax) { aMax = imu.accMag[i]; iAcc = i; }

            if (aMax < o.swingMinPeakG) continue;

            swings.push({
                index: swings.length,
                iStart: s, iPeak: p, iEnd: e, iImpact: iAcc,
                tStart: imu.t[s], tPeak: imu.t[p], tEnd: imu.t[e], tImpact: imu.t[iAcc],
                peakGyroDps: env[p], rawPeakGyroDps: imu.gyrMag[p], peakAccG: aMax
            });
        }
        return swings;
    }

    // ------------------------------------------------------------ quaternion --

    function qMul(a, b) {
        return [
            a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]
        ];
    }
    function qNorm(q) {
        var n = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
        return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
    }
    function qConj(q) { return [q[0], -q[1], -q[2], -q[3]]; }
    /** Rotate vector v by quaternion q. */
    function qRot(q, v) {
        var r = qMul(qMul(q, [0, v[0], v[1], v[2]]), qConj(q));
        return [r[1], r[2], r[3]];
    }
    /** Shortest rotation taking unit vector a onto unit vector b. */
    function qFromVectors(a, b) {
        var d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        if (d > 0.999999) return [1, 0, 0, 0];
        if (d < -0.999999) {
            // 180 deg: any axis perpendicular to a
            var ax = Math.abs(a[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
            var c = [a[1] * ax[2] - a[2] * ax[1], a[2] * ax[0] - a[0] * ax[2], a[0] * ax[1] - a[1] * ax[0]];
            var n = Math.hypot(c[0], c[1], c[2]) || 1;
            return [0, c[0] / n, c[1] / n, c[2] / n];
        }
        var cx = a[1] * b[2] - a[2] * b[1], cy = a[2] * b[0] - a[0] * b[2], cz = a[0] * b[1] - a[1] * b[0];
        return qNorm([1 + d, cx, cy, cz]);
    }
    /** Total rotation angle of a quaternion, in degrees. */
    function qAngleDeg(q) {
        var w = Math.min(1, Math.max(-1, Math.abs(q[0])));
        return 2 * Math.acos(w) / DEG;
    }

    // ---------------------------------------------- orientation over a swing --

    /**
     * Racket orientation through one swing, integrated from the gyroscope.
     *
     * Why this shape: a 6-axis IMU (MPU6050 -- accelerometer + gyroscope, NO
     * magnetometer) can hold roll and pitch indefinitely by leaning on gravity,
     * but has nothing to correct heading (yaw) against, so free-running yaw
     * drifts without bound. During a swing the accelerometer reads gravity PLUS
     * up to ~11 g of swing acceleration, so it cannot correct anything either.
     *
     * So: take the absolute attitude from gravity during the quiet ready
     * position just before the swing (where |a| ~ 1 g and rotation is low),
     * then integrate the gyro forward across the swing only -- a window of a
     * few hundred milliseconds, where integration drift stays small. Yaw is
     * defined as 0 at that reference, making every reported angle a change
     * FROM the player's own ready position rather than an absolute compass
     * heading. That is the honest quantity this hardware can produce, and it
     * is the one that matters for "did I close the face more than last time".
     */
    function orientationThroughSwing(imu, swing, opts) {
        var o = Object.assign({
            readyWindowS: 0.20,      // length of the "ready position" reference window
            readySearchS: 0.70,      // how far back from the swing to look for it
            maxIntegrationS: 1.00,   // hard cap: past this, gyro drift dominates
            quietGyroDps: 150
        }, opts || {});
        var fs = imu.fs;
        var win = Math.max(3, Math.round(o.readyWindowS * fs));
        var search = Math.max(win, Math.round(o.readySearchS * fs));

        // --- find the quietest window in the run-up, preferring the LATEST one ---
        // Later is better: every extra second between the reference and contact is
        // another second of gyro integration drift.
        // Bound the search so the reference can never sit further from contact
        // than the integration budget allows.
        var lo = Math.max(0, swing.iStart - search,
            swing.iPeak - Math.round(o.maxIntegrationS * fs) - win);
        var hi = swing.iStart;
        if (lo + win > hi) lo = Math.max(0, hi - win);
        var bestI = -1, bestScore = Infinity;
        for (var s = lo; s + win <= hi; s++) {
            var gsum = 0, adev = 0;
            for (var i = s; i < s + win; i++) {
                gsum += imu.gyrMag[i];
                adev += Math.abs(imu.accMag[i] - 1);
            }
            var recency = (hi - (s + win)) / fs;             // seconds before the swing starts
            var score = gsum / win + 200 * (adev / win) + 60 * recency;
            if (score < bestScore) { bestScore = score; bestI = s; }
        }
        if (bestI < 0) return null;

        var meanGyro = 0, gAx = 0, gAy = 0, gAz = 0, gVar = 0;
        var j;
        for (j = bestI; j < bestI + win; j++) {
            meanGyro += imu.gyrMag[j] / win;
            gAx += imu.ax[j] / win; gAy += imu.ay[j] / win; gAz += imu.az[j] / win;
        }
        var gMag = Math.hypot(gAx, gAy, gAz);
        var referenceOk = meanGyro < o.quietGyroDps && Math.abs(gMag - 1) < 0.25;

        // --- gyro bias from that same quiet window (removes most of the drift) ---
        var bx = 0, by = 0, bz = 0;
        for (var b = bestI; b < bestI + win; b++) { bx += imu.gx[b] / win; by += imu.gy[b] / win; bz += imu.gz[b] / win; }
        for (j = bestI; j < bestI + win; j++) {
            var dx = imu.gx[j] - bx, dy = imu.gy[j] - by, dz = imu.gz[j] - bz;
            gVar += (dx * dx + dy * dy + dz * dz) / win;
        }
        var biasNoiseDps = Math.sqrt(gVar / win);            // s.e.m. of the bias estimate

        // --- absolute roll/pitch from gravity; yaw defined as 0 here ---
        var gUnit = gMag > 0 ? [gAx / gMag, gAy / gMag, gAz / gMag] : [0, 0, 1];
        // q0 maps the sensor's measured gravity direction onto world "up" (+Z).
        var q = qFromVectors(gUnit, [0, 0, 1]);

        // --- integrate the gyro from the reference across the swing ---
        var start = bestI + win - 1;
        var stopIdx = Math.min(swing.iEnd, imu.n - 1,
            start + Math.round(o.maxIntegrationS * fs));
        // Contact is the measurement that matters -- never stop short of it.
        stopIdx = Math.max(stopIdx, Math.min(swing.iPeak, imu.n - 1));

        var quats = [], idxs = [];
        for (var k = start; k <= stopIdx; k++) {
            if (k > start) {
                var dt = imu.t[k] - imu.t[k - 1];
                if (!(dt > 0) || dt > 0.1) dt = 1 / fs;      // guard a stalled/absurd timestamp
                var wx = (imu.gx[k] - bx) * DEG, wy = (imu.gy[k] - by) * DEG, wz = (imu.gz[k] - bz) * DEG;
                var half = 0.5 * dt;
                var dq = [1, wx * half, wy * half, wz * half];
                q = qNorm(qMul(q, dq));
            }
            quats.push(q); idxs.push(k);
        }

        var spanToContact = imu.t[Math.min(swing.iPeak, stopIdx)] - imu.t[start];
        return {
            referenceIndex: bestI, referenceEnd: start,
            referenceOk: referenceOk,
            referenceGyroDps: meanGyro, referenceAccG: gMag,
            // Unit gravity direction in the sensor frame at the quiet reference
            // (this IS world "up" there). Exposed so a later step can measure the
            // swing's rotation about world-vertical without any magnetometer.
            gUnit: gUnit,
            gyroBiasDps: [bx, by, bz], biasNoiseDps: biasNoiseDps,
            indices: idxs, quats: quats,
            integrationSeconds: spanToContact,
            // Drift budget: an unmodelled bias of biasNoiseDps integrated over the
            // window. Reported so a face angle is never quoted more precisely
            // than the sensor can support.
            driftBoundDeg: biasNoiseDps * Math.max(0, spanToContact)
        };
    }

    /**
     * Racket-face geometry at contact, expressed as change from the ready
     * position. Reported components:
     *
     *   faceRotationDeg  -- total 3-D rotation of the racket between ready and
     *                       contact (orientation-convention free: it does not
     *                       depend on how the sensor happens to sit on the frame)
     *   closeOpenDeg     -- rotation about the racket's own long axis, the
     *                       component a coach calls "closing"/"opening" the face
     *   tiltDeg          -- rotation of the long axis away from its ready
     *                       direction (how much the racket head dropped/lifted)
     *
     * Without a mounting calibration the sensor's axes are not known relative to
     * the frame, so `longAxis` defaults to the axis that rotated LEAST over the
     * take -- for a racket swing that is the shaft direction, since a swing
     * rotates the frame mostly about its own handle and about the vertical.
     * Pass opts.longAxis ([x,y,z] in sensor frame) once you calibrate, and these
     * become exact rather than inferred.
     */
    function faceGeometry(imu, swing, orient, opts) {
        var o = Object.assign({ longAxis: null }, opts || {});
        if (!orient || !orient.quats.length) return null;

        var qRef = orient.quats[0];
        var iContact = swing.iPeak;
        var pos = orient.indices.indexOf(iContact);
        if (pos < 0) pos = orient.indices.length - 1;
        var qC = orient.quats[pos];

        // rotation taking the ready attitude to the contact attitude
        var dq = qMul(qConj(qRef), qC);
        var faceRotationDeg = qAngleDeg(dq);

        // Rotation about world-vertical (gravity), in degrees -- the quantity
        // that flips sign between a forehand and a backhand for the same player.
        // Derived from the quaternion delta projected onto the reference gravity
        // direction, so it is mount-independent (no fixed sensor axis assumed).
        var vrot = 0;
        if (orient.gUnit) {
            var axx = dq[1], ayy = dq[2], azz = dq[3];
            var an = Math.hypot(axx, ayy, azz);
            if (an > 1e-9) {
                var ang = 2 * Math.acos(Math.min(1, Math.abs(dq[0]))) / DEG; // deg
                var ux = axx / an, uy = ayy / an, uz = azz / an;
                var dotu = ux * orient.gUnit[0] + uy * orient.gUnit[1] + uz * orient.gUnit[2];
                vrot = ang * dotu;
            }
        }

        var axis = o.longAxis;
        if (!axis) {
            // Infer the shaft: the sensor axis whose direction moves least
            // between ready and contact.
            var cands = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
            var best = null;
            for (var c = 0; c < 3; c++) {
                var v = qRot(dq, cands[c]);
                var dot = v[0] * cands[c][0] + v[1] * cands[c][1] + v[2] * cands[c][2];
                if (best === null || dot > best.dot) best = { dot: dot, axis: cands[c] };
            }
            axis = best.axis;
        }
        var an = Math.hypot(axis[0], axis[1], axis[2]) || 1;
        axis = [axis[0] / an, axis[1] / an, axis[2] / an];

        var moved = qRot(dq, axis);
        var d = Math.min(1, Math.max(-1, moved[0] * axis[0] + moved[1] * axis[1] + moved[2] * axis[2]));
        var tiltDeg = Math.acos(d) / DEG;

        // twist about the long axis = swing decomposition of dq onto that axis
        var proj = dq[1] * axis[0] + dq[2] * axis[1] + dq[3] * axis[2];
        var twist = qNorm([dq[0], proj * axis[0], proj * axis[1], proj * axis[2]]);
        var closeOpenDeg = qAngleDeg(twist) * (proj < 0 ? -1 : 1);

        return {
            faceRotationDeg: faceRotationDeg,
            verticalRotDeg: vrot,
            closeOpenDeg: closeOpenDeg,
            tiltDeg: tiltDeg,
            longAxis: axis,
            calibrated: !!o.longAxis,
            referenceOk: orient.referenceOk,
            referenceGyroDps: orient.referenceGyroDps,
            integrationSeconds: orient.integrationSeconds,
            uncertaintyDeg: orient.driftBoundDeg,
            // Trustworthy when gravity gave a valid attitude to start from and
            // the drift budget over the integration window stayed small. A
            // restless ready position is reported (referenceGyroDps) but is not
            // by itself disqualifying -- players rarely stand still between
            // reps, and what actually limits the measurement is the drift bound.
            trustworthy: Math.abs(orient.referenceAccG - 1) < 0.30
                && orient.integrationSeconds <= 1.05
                && orient.driftBoundDeg <= 8
        };
    }

    // ------------------------------------------------------- per-swing physics --

    /**
     * Racket-head speed from the measured angular rate.
     *
     * v = omega x r. Using the gyro directly is far more robust than double-
     * integrating the accelerometer (whose error grows with t^2 and which reads
     * gravity plus swing acceleration mixed together). `radiusM` is the distance
     * from the rotation centre to the racket head; the default assumes a swing
     * pivoting roughly about the shoulder with the head ~1.0 m out. It is a
     * configurable geometric constant, not a measurement -- the number is an
     * estimate whose *relative* changes between swings are meaningful even where
     * its absolute value is only approximate.
     */
    function headSpeed(peakGyroDps, radiusM) {
        var r = radiusM || 1.0;
        var mps = (peakGyroDps * DEG) * r;
        return { mps: mps, kmh: mps * 3.6, radiusM: r };
    }

    /** Swing tempo, resolved at the IMU's own sample rate (5 ms at 200 Hz) --
     *  finer than any 30 fps camera can measure. */
    function tempo(imu, swing) {
        return {
            backswingMs: (swing.tPeak - swing.tStart) * 1000,
            followThroughMs: (swing.tEnd - swing.tPeak) * 1000,
            totalMs: (swing.tEnd - swing.tStart) * 1000,
            ratio: (swing.tEnd - swing.tPeak) > 0
                ? (swing.tPeak - swing.tStart) / (swing.tEnd - swing.tPeak) : NaN
        };
    }

    /**
     * Post-contact vibration ("was it hit off-centre?").
     *
     * A clean strike damps smoothly; an off-centre hit rings the frame. This
     * high-passes |accel| just after the impact index and reports its RMS.
     *
     * TWO limits, and both matter:
     *
     * 1. It is meaningless without a ball. There is nothing to strike in a
     *    shadow swing, so what this measures is the racket's own deceleration,
     *    not an impact. `applicable` is false unless the caller declares ball
     *    practice (`opts.withBall`), because the sensor genuinely cannot tell.
     *    An earlier version tried to infer ball contact from peak acceleration
     *    plus ring energy, and on the project's own ball-free recordings it
     *    called 4 of 5 forehands a ball strike -- a hard swing alone clears any
     *    such threshold. Guessing here was worse than declining to guess.
     * 2. Even with a ball, 200 Hz sampling caps observable content at 100 Hz
     *    (Nyquist), while real string-bed vibration lives in the high hundreds.
     *    What survives is the ENVELOPE of the impact shock -- a relative
     *    indicator for comparing one strike against another on the same racket,
     *    never an absolute measure of where on the string bed the ball landed.
     */
    function impactSignature(imu, swing, opts) {
        var o = Object.assign({ windowS: 0.15, hpCutoffHz: 25, withBall: false }, opts || {});
        var i0 = swing.iImpact;
        var n = Math.max(4, Math.round(o.windowS * imu.fs));
        var hi = Math.min(imu.n, i0 + n);

        // one-pole high-pass on |accel| over the post-impact window
        var alpha = 1 / (1 + 2 * Math.PI * o.hpCutoffHz / imu.fs);
        var prevIn = imu.accMag[i0], prevOut = 0, sum = 0, cnt = 0, peak = 0;
        for (var i = i0 + 1; i < hi; i++) {
            var x = imu.accMag[i];
            var y = alpha * (prevOut + x - prevIn);
            prevIn = x; prevOut = y;
            sum += y * y; cnt++;
            if (Math.abs(y) > peak) peak = Math.abs(y);
        }
        var rms = cnt ? Math.sqrt(sum / cnt) : 0;

        // jerk = d|a|/dt at the impact, in g/s
        var jerk = 0;
        for (var k = Math.max(1, i0 - 2); k < Math.min(imu.n, i0 + 3); k++) {
            var dt = imu.t[k] - imu.t[k - 1] || 1 / imu.fs;
            var j = Math.abs(imu.accMag[k] - imu.accMag[k - 1]) / dt;
            if (j > jerk) jerk = j;
        }

        return {
            ringRmsG: rms, ringPeakG: peak, peakJerkGps: jerk,
            applicable: !!o.withBall,
            note: o.withBall
                ? 'Relative indicator only -- 200 Hz sampling sees the shock envelope, not string vibration.'
                : 'Not applicable: shadow swing (no ball declared). With no ball there is no impact to characterise.'
        };
    }

    /**
     * Swing energy. Turns "peak acceleration" into something a player can
     * picture. Uses the gyro-derived head speed above, with a configurable
     * racket mass (default 300 g, a common strung frame).
     */
    function swingEnergy(peakGyroDps, opts) {
        var o = Object.assign({ racketMassKg: 0.30, radiusM: 1.0 }, opts || {});
        var hs = headSpeed(peakGyroDps, o.radiusM);
        return {
            headSpeedKmh: hs.kmh, headSpeedMps: hs.mps,
            kineticEnergyJ: 0.5 * o.racketMassKg * hs.mps * hs.mps,
            racketMassKg: o.racketMassKg, radiusM: o.radiusM
        };
    }

    /**
     * Everything for one swing, in one object.
     * `opts.stroke` (e.g. 'forehand_volley') gates the stroke-specific rules.
     */
    function analyzeSwing(imu, swing, opts) {
        var o = opts || {};
        var orient = orientationThroughSwing(imu, swing, o);
        var face = faceGeometry(imu, swing, orient, o);
        // Total angular travel across the whole swing window (ready -> follow-through),
        // in degrees. A volley is a compact punch with almost no backswing, so its arc
        // is small; a groundstroke sweeps a large arc (backswing + follow-through).
        // This arc, not peak rate, is what most cleanly tells a forehand from a volley
        // when both are swung quickly.
        var arcDeg = 0;
        for (var ai = swing.iStart; ai <= swing.iEnd; ai++) arcDeg += imu.gyrMag[ai];
        arcDeg /= imu.fs;   // sum of deg/s over samples -> degrees (gyrMag already in deg/s)
        return {
            index: swing.index,
            tStart: swing.tStart, tContact: swing.tPeak, tEnd: swing.tEnd,
            iStart: swing.iStart, iContact: swing.iPeak, iEnd: swing.iEnd,
            peakGyroDps: swing.peakGyroDps,
            peakAccG: swing.peakAccG,
            peakAccMps2: swing.peakAccG * G,
            arcDeg: arcDeg,
            face: face,
            tempo: tempo(imu, swing),
            impact: impactSignature(imu, swing, o),
            energy: swingEnergy(swing.peakGyroDps, o)
        };
    }

    /**
     * Pick ONE racket long-axis for the whole take by majority vote across its
     * swings. Doing this per-swing lets the inferred axis flip between reps,
     * which would flip the sign of closeOpenDeg and make the session-level
     * consistency numbers meaningless. One axis per take keeps "closing" and
     * "opening" pointing the same way for every rep the player compares.
     */
    function inferLongAxis(imu, swings, opts) {
        var votes = [0, 0, 0], cands = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
        for (var s = 0; s < swings.length; s++) {
            var orient = orientationThroughSwing(imu, swings[s], opts);
            if (!orient || !orient.quats.length) continue;
            var pos = orient.indices.indexOf(swings[s].iPeak);
            if (pos < 0) pos = orient.quats.length - 1;
            var dq = qMul(qConj(orient.quats[0]), orient.quats[pos]);
            var best = -Infinity, bestC = 0;
            for (var c = 0; c < 3; c++) {
                var v = qRot(dq, cands[c]);
                var dot = v[0] * cands[c][0] + v[1] * cands[c][1] + v[2] * cands[c][2];
                if (dot > best) { best = dot; bestC = c; }
            }
            votes[bestC]++;
        }
        var win = votes[0] >= votes[1] ? (votes[0] >= votes[2] ? 0 : 2) : (votes[1] >= votes[2] ? 1 : 2);
        return { axis: cands[win], votes: votes, unanimous: votes[win] === swings.length };
    }

    /** Analyze a whole take: parse -> taps -> swings -> per-swing metrics. */
    function analyzeTake(imu, opts) {
        var o = Object.assign({}, opts || {});
        var taps = detectTaps(imu, o);
        var swings = detectSwings(imu, o, taps);
        var axisInfo = null;
        if (!o.longAxis && swings.length) {
            axisInfo = inferLongAxis(imu, swings, o);
            o = Object.assign({}, o, { longAxis: axisInfo.axis, longAxisInferred: true });
        }
        var analyzed = swings.map(function (s) { return analyzeSwing(imu, s, o); });
        // `calibrated` must stay false when the axis was inferred rather than measured.
        if (axisInfo) analyzed.forEach(function (a) { if (a.face) a.face.calibrated = false; });
        return { imu: imu, taps: taps, swings: swings, analyzed: analyzed, axisInfo: axisInfo };
    }

    // -------------------------------------------------------- published rules --

    // Aydin, E.H. & Aydemir, O. (2026). Sensors 26(10), 3273. Peak dominant-hand
    // acceleration for volleys, (mean, sd) in m/s^2. Elite volleys are LOWER --
    // "amateurs overcompensate for technical deficiencies with excessive,
    // uncontrolled force". Mirrors ml/skilleye/quality/skill_rules.py so the
    // browser and the Python backend cannot drift apart on the numbers.
    var AYDIN_AYDEMIR_VOLLEY = { elite: [48.12, 26.49], amateur: [57.09, 29.86] };

    /**
     * Volley swing effort (Aydin & Aydemir 2026).
     *
     * Flags when peak acceleration passes the midpoint between the two
     * published group means. Returns null -- deliberately -- unless the caller
     * declares BOTH that this is a volley and that a ball was actually struck,
     * because two separate mismatches make the comparison invalid otherwise:
     *
     * 1. NO BALL. The paper measured volleys played against a real ball. A
     *    shadow swing has no collision, so its peak is pure swing dynamics.
     * 2. DIFFERENT SENSOR POSITION. The paper measured DOMINANT-HAND
     *    acceleration. This sensor sits at the racket throat, further from the
     *    rotation centre, and centripetal acceleration scales with radius
     *    (a = omega^2 * r) -- so it reads systematically higher than the
     *    quantity the thresholds were built from.
     *
     * Both push the same way, and the effect is not subtle: on the project's
     * own ball-free recordings every one of the ten strokes cleared the
     * "amateur" midpoint (53-110 m/s^2 against a 52.6 midpoint). Wired
     * ungated, this rule would tell every user on every stroke that they swing
     * too hard -- a constant, and a constant carries no information.
     *
     * `opts.withBall` is the user's own declaration; there is no way to detect
     * ball contact from a 200 Hz stream (see impactSignature). Even when it is
     * true, `mountingCaveat` stays set, because point 2 is unfixed until the
     * sensor is calibrated against a hand-mounted reference.
     */
    function checkVolleySwingEffort(peakAccelMps2, stroke, opts) {
        var o = Object.assign({ withBall: false }, opts || {});
        if (stroke !== 'forehand_volley' && stroke !== 'backhand_volley') return null;
        if (!o.withBall) {
            return {
                applicable: false, reason: 'no-ball',
                zh: '此規則需要實際擊球。無球揮拍沒有碰撞，其峰值加速度與論文量測的量不同，因此不予評估。',
                en: 'This rule needs a real ball strike. A shadow swing has no collision, so its peak '
                    + 'acceleration is not the quantity the paper measured -- not evaluated.'
            };
        }
        var mid = (AYDIN_AYDEMIR_VOLLEY.elite[0] + AYDIN_AYDEMIR_VOLLEY.amateur[0]) / 2;
        var flagged = peakAccelMps2 > mid;
        var caveat = {
            zh: '注意：論文量測的是「慣用手」的加速度，本感測器裝在拍喉，離旋轉中心更遠，'
                + '讀值系統性偏高（a = ω²r）。在完成對照校準前，此判定僅供參考。',
            en: 'Caveat: the paper measured DOMINANT-HAND acceleration; this sensor sits at the racket '
                + 'throat, further from the rotation centre, so it reads systematically higher '
                + '(a = omega^2 * r). Treat this as indicative until the mounting is calibrated.'
        };
        return {
            applicable: true, flagged: flagged, valueMps2: peakAccelMps2, midpointMps2: mid,
            mountingCaveat: caveat,
            zh: (flagged
                ? '本次截擊的揮拍力道（' + peakAccelMps2.toFixed(1) + ' m/s²）偏向高出力型態'
                  + '（參考中點 ' + mid.toFixed(1) + ' m/s²）。Aydin & Aydemir (2026) 發現優秀選手的截擊揮拍更短、'
                  + '更受控——請嘗試縮短揮拍幅度，而不是加大力量。'
                : '本次截擊的揮拍力道（' + peakAccelMps2.toFixed(1) + ' m/s²）落在受控範圍內，請保持這個手感。')
                + ' ' + caveat.zh,
            en: (flagged
                ? 'Volley swing effort (' + peakAccelMps2.toFixed(1) + ' m/s²) leans toward the higher-force '
                  + 'pattern (reference midpoint ' + mid.toFixed(1) + ' m/s²). Try a shorter, more compact swing '
                  + 'rather than swinging harder.'
                : 'Volley swing effort (' + peakAccelMps2.toFixed(1) + ' m/s²) sits in the controlled range. Keep this feel.')
                + ' ' + caveat.en
        };
    }

    /**
     * Stroke recognition from the racket IMU alone -- so "with sensor" mode can
     * report a stroke type the camera never saw. Everything below is a rule on
     * MEASURED swing features; there is no fixed label anywhere. Thresholds are
     * physical (swing duration, peak swing acceleration, rotation sense about
     * world-vertical) and tunable. There is no ball in this project, so the
     * "acceleration" used here is the racket's OWN swing acceleration, never an
     * impact. Derived and validated against the project's own labelled recordings
     * (hardware/client/newresult/*.csv): volleys are compact, serves are short with
     * very high swing acceleration (the overhead serve whips the racket hard even
     * with nothing struck), smashes are
     * long overhead swings, and forehand vs backhand flips the sign of the
     * rotation about gravity.
     *
     * Returns null when no swing was detected (caller should fall back to the
     * video model), so a quiet or borderline log never produces a fake label.
     */
    function classifyStroke(imu, take, opts) {
        var o = Object.assign({
            handedness: 'right',          // which vertical-rotation sense is "forehand"
            volleyMaxMs: 700,             // shorter than this + modest accel => volley
            groundstrokeMinMs: 1100,      // longer than this => smash (overhead)
            servePeakAccelG: 9,           // short AND this hard a swing => serve
                                        // (racket's own acceleration; no ball struck)
            // A volley is a compact BLOCK: short, low accel, almost no topspin (roll),
            // low racket-head speed, and a small total swing arc. A compact forehand
            // groundstroke shares the short duration and low accel, so it must be
            // separated on the OTHER axes -- roll (a groundstroke brushes up), peak
            // rate, and total arc (a groundstroke still has a backswing + follow-through).
            // The volley gate therefore requires ALL of these, so a quick forehand with
            // any real topspin or arc is filed as a groundstroke and only a clear punch
            // is called a volley. Tuned on the project's real recordings
            // (ml/results/imu_recordings): genuine volleys sit at arc<~160deg,
            // roll<~60deg, peakG<~300dps; compact forehands are higher on at least one.
            volleyMaxAccelG: 5,
            volleyMaxRollDeg: 70,         // max long-axis twist (topspin) for a volley
            volleyMaxGyroDps: 500,        // max peak angular rate (head speed) for a volley
            volleyMaxArcDeg: 180          // max total swing arc (deg) for a volley
        }, opts || {});
        if (!take || !take.analyzed || !take.analyzed.length) return null;

        // Representative swing = the one with the most rotation (strongest hit).
        var swings = take.analyzed.slice().sort(function (a, b) {
            return b.peakGyroDps - a.peakGyroDps;
        });
        var rep = swings[0];
        var dur = rep.tempo.totalMs;
        var peakA = rep.peakAccG;
        var peakG = rep.peakGyroDps;
        var vRot = (rep.face && isFinite(rep.face.verticalRotDeg)) ? rep.face.verticalRotDeg : 0;
        // Long-axis twist (closeOpenDeg) = how much the racket face rolled (topspin).
        // A volley blocks; a groundstroke brushes up, so groundstrokes roll far more.
        var roll = (rep.face && isFinite(rep.face.closeOpenDeg)) ? Math.abs(rep.face.closeOpenDeg) : 0;
        // Total swing arc (deg) -- backswing + forward + follow-through. A volley is a
        // tiny punch; a groundstroke sweeps a large arc even when swung quickly.
        var arc = isFinite(rep.arcDeg) ? rep.arcDeg : 0;

        // Forehand/backhand sense, with cross-swing agreement for confidence.
        var signs = swings.map(function (s) {
            return (s.face && isFinite(s.face.verticalRotDeg)) ? Math.sign(s.face.verticalRotDeg) : 0;
        }).filter(function (x) { return x !== 0; });
        var pos = signs.filter(function (x) { return x > 0; }).length;
        var neg = signs.length - pos;
        var agree = signs.length ? Math.max(pos, neg) / signs.length : 0.5;
        var fhSign = (o.handedness === 'right') ? 1 : -1;
        var isForehand = (vRot * fhSign) >= 0;

        // Family decision (overhead serve / smash / volley / groundstroke).
        var family, fMargin;
        if (peakA >= o.servePeakAccelG && dur < o.groundstrokeMinMs) {
            family = 'serve';
            fMargin = (peakA - o.servePeakAccelG) / o.servePeakAccelG;
        } else if (dur >= o.groundstrokeMinMs) {
            family = 'smash';
            fMargin = (dur - o.groundstrokeMinMs) / o.groundstrokeMinMs;
        } else if (dur < o.volleyMaxMs && peakA < o.volleyMaxAccelG
                   && roll < o.volleyMaxRollDeg && peakG < o.volleyMaxGyroDps
                   && arc < o.volleyMaxArcDeg) {
            family = 'volley';
            fMargin = 1 - Math.min(1, Math.abs(dur - o.volleyMaxMs / 2) / (o.volleyMaxMs / 2));
        } else {
            family = 'groundstroke';
            fMargin = 0.4;
        }

        var stroke;
        if (family === 'serve') stroke = 'serve';
        else if (family === 'smash') stroke = 'smash';
        else if (family === 'volley') stroke = isForehand ? 'forehand_volley' : 'backhand_volley';
        else stroke = isForehand ? 'forehand' : 'backhand';

        // Confidence blends how decisively the family was chosen with how
        // consistently every swing agreed on the forehand/backhand sense.
        var confidence = Math.max(0, Math.min(1, 0.6 * Math.min(1, fMargin + 0.3) + 0.4 * agree));
        return {
            stroke: stroke, confidence: confidence, family: family,
            handedness: o.handedness,
            features: { durationMs: dur, peakAccelG: peakA, peakGyroDps: peakG,
                        verticalRotDeg: vRot, rollDeg: roll, arcDeg: arc,
                        forehandSignAgreement: agree },
            method: 'imu-feature-rules'
        };
    }

    return {
        G: G,
        parseImuCsv: parseImuCsv,
        detectTaps: detectTaps,
        detectSwings: detectSwings,
        orientationThroughSwing: orientationThroughSwing,
        faceGeometry: faceGeometry,
        headSpeed: headSpeed,
        tempo: tempo,
        impactSignature: impactSignature,
        swingEnergy: swingEnergy,
        analyzeSwing: analyzeSwing,
        analyzeTake: analyzeTake,
        inferLongAxis: inferLongAxis,
        classifyStroke: classifyStroke,
        checkVolleySwingEffort: checkVolleySwingEffort,
        AYDIN_AYDEMIR_VOLLEY: AYDIN_AYDEMIR_VOLLEY,
        DEFAULTS: DEFAULTS,
        util: {
            smooth: smooth, median: median, percentile: percentile,
            mean: mean, stdev: stdev, cv: cv, findPeaks: findPeaks,
            qMul: qMul, qRot: qRot, qConj: qConj, qNorm: qNorm,
            qFromVectors: qFromVectors, qAngleDeg: qAngleDeg
        }
    };
}));
