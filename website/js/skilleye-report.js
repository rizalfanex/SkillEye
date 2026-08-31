/*!
 * skilleye-report.js -- shared HTML rendering for the sensor reports.
 *
 * Used by analysis.html (single take) and session.html (whole session) so the
 * two pages cannot drift apart on wording, units, or how uncertainty is shown.
 *
 * One rule runs through all of it: never print a number more precisely than it
 * was measured. Racket-face angles always carry their drift budget, estimated
 * quantities are labelled as estimates, and anything the sensor could not
 * establish says so instead of showing a plausible-looking blank.
 *
 * Loads as a browser global (window.SkillEyeReport).
 */
(function (root, factory) {
    if (typeof module === 'object' && module.exports) module.exports = factory();
    else root.SkillEyeReport = factory();
}(typeof self !== 'undefined' ? self : this, function () {
    'use strict';

    function esc(s) {
        return String(s).replace(/[&<>"']/g, function (c) {
            return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
        });
    }

    /** Bilingual span pair matching the site's existing .lang-zh / .lang-en scheme. */
    function bi(zh, en) {
        return '<span class="lang-zh">' + esc(zh) + '</span>'
            + '<span class="lang-en lang-hidden">' + esc(en) + '</span>';
    }

    function num(v, digits, fallback) {
        return (typeof v === 'number' && isFinite(v)) ? v.toFixed(digits == null ? 1 : digits)
            : (fallback == null ? '—' : fallback);
    }

    var LEVEL_COLORS = {
        good: '#2d6a4f', excellent: '#2d6a4f', fair: '#d97706',
        poor: '#c0392b', unknown: '#888'
    };

    // ------------------------------------------------------------ sync card --

    function syncCard(sync) {
        if (!sync || !sync.ok) {
            return '<div class="se-card se-card-warn">'
                + '<h4>' + bi('⚠️ 無法對齊攝影機與感測器',
                    '⚠️ Could not align camera and sensor') + '</h4>'
                + '<p>' + bi(
                    (sync && sync.advice) ? sync.advice : '事件不足，無法對齊。',
                    (sync && sync.advice) ? sync.advice : 'Not enough events to align.') + '</p>'
                + '</div>';
        }

        var color = LEVEL_COLORS[sync.quality.level] || '#888';
        var xc = sync.crossCheck;

        return '<div class="se-card" style="border-left-color:' + color + ';">'
            + '<h4>' + bi('🔗 攝影機 ↔ 感測器對齊', '🔗 Camera ↔ sensor alignment') + '</h4>'
            + '<div class="se-kpis">'
            + kpi(bi('時間偏移', 'Time offset'), num(sync.offset, 3) + ' s')
            + kpi(bi('配對成功球數', 'Strokes matched'), sync.inliers + ' / ' + sync.imuEvents)
            + kpi(bi('殘差', 'Residual'), num(sync.residualStdMs, 1) + ' ms')
            + kpi(bi('單一影格', 'One frame'), num(sync.frameMs, 1) + ' ms')
            + '</div>'
            + (sync.videoCoverage != null && sync.videoCoverage < 0.9
                ? '<p class="se-note">' + bi(
                    '瀏覽器只取到 ' + Math.round(sync.videoCoverage * 100) + '% 的影格'
                    + '（取樣間隔 ' + num(sync.sampledFrameMs, 0) + ' ms），因為影片被加速播放以節省時間。'
                    + '上方的殘差仍以攝影機真正的影格間隔為基準，而非這個較稀疏的取樣率。',
                    'The browser sampled only ' + Math.round(sync.videoCoverage * 100) + '% of frames '
                    + '(' + num(sync.sampledFrameMs, 0) + ' ms apart) because the video was played faster to save time. '
                    + 'The residual above is still judged against the camera\'s real frame interval, not this sparser sampling.')
                + '</p>' : '')
            + '<p class="se-verdict" style="color:' + color + ';">' + bi(sync.quality.zh, sync.quality.en) + '</p>'
            + '<p class="se-note">' + bi(
                sync.subFrame
                    ? '對齊誤差小於一個影格——這已是 30 fps 攝影機的解析極限，'
                      + '除非提高影格率，否則無法更精確。'
                    : '對齊誤差超過一個影格。結果仍可使用，但把握度較低。',
                sync.subFrame
                    ? 'Alignment error is below one video frame -- the resolution limit of a 30 fps camera, '
                      + 'not improvable without a faster camera.'
                    : 'Alignment error exceeds one video frame. Still usable, but less certain.') + '</p>'
            + (xc ? '<p class="se-note">' + bi(
                '全波形相關性交叉驗證：' + num(xc.offset, 3) + ' s（'
                + (xc.deltaMs > 0 ? '+' : '') + num(xc.deltaMs, 0) + ' ms = ' + num(xc.deltaFrames, 1)
                + ' 影格，相關係數 ' + num(xc.correlation, 2) + '）。'
                + (xc.agrees ? '兩種獨立方法得到一致結果。'
                    : '兩種方法結果不一致——建議重看影片。'),
                'Independent whole-waveform cross-check: ' + num(xc.offset, 3) + ' s ('
                + (xc.deltaMs > 0 ? '+' : '') + num(xc.deltaMs, 0) + ' ms = ' + num(xc.deltaFrames, 1)
                + ' frames, correlation ' + num(xc.correlation, 2) + '). '
                + (xc.agrees ? 'Two independent methods agree.'
                    : 'The two methods disagree -- worth re-checking the video.')) + '</p>' : '')
            + (sync.clockScale && sync.clockScale.justified
                ? '<p class="se-note">' + bi(
                    '偵測到明顯的時鐘漂移（' + num(sync.clockScale.ppm, 0)
                    + ' ppm）——長時間錄影建議啟用比例校正。',
                    'Significant clock drift detected (' + num(sync.clockScale.ppm, 0)
                    + ' ppm) -- for long takes, enable scale correction.') + '</p>'
                : '<p class="se-note">' + bi(
                    '本次錄影不需要時鐘比例校正（僅使用固定偏移）。',
                    'No clock-scale correction needed for this take (constant offset only).') + '</p>')
            + '</div>';
    }

    /**
     * Shown when the camera half could not run. Deliberately separates "the
     * alignment was skipped" from "the analysis failed": the sensor metrics
     * below it are complete either way, and a user who sees a red failure
     * banner will not read the working report underneath it.
     */
    function videoIssueCard(issue) {
        if (!issue) return '';
        var isNoVideo = issue.reason === 'no-video';
        var html = '<div class="se-card' + (isNoVideo ? '' : ' se-card-warn') + '">'
            + '<h4>' + (isNoVideo
                ? bi('ℹ️ 未進行攝影機對齊（無影片）', 'ℹ️ No camera alignment (no video)')
                : bi('⚠️ 影片無法用於對齊', '⚠️ Video unusable for alignment')) + '</h4>'
            + '<p style="font-size:0.84rem;color:#374151;line-height:1.6;">'
            + bi(issue.zh || '', issue.en || '') + '</p>';
        if (issue.fixZh || issue.fixEn) {
            html += '<p class="se-note" style="background:#fff;padding:10px 12px;border-radius:10px;'
                + 'border:1px solid #fed7aa;margin-top:10px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;'
                + 'font-size:0.72rem;white-space:pre-wrap;">'
                + bi(issue.fixZh || '', issue.fixEn || '') + '</p>';
        }
        html += '<p class="se-note">' + bi(
            '下方的感測器數據不依賴攝影機，仍然有效——'
            + '只有「把擊球瞬間對應到影片影格」這件事無法進行。',
            'The sensor metrics below do not depend on the camera and remain valid -- '
            + 'only mapping contact instants onto video frames is unavailable.') + '</p>';
        return html + '</div>';
    }

    function kpi(label, value) {
        return '<div class="se-kpi"><div class="se-kpi-label">' + label + '</div>'
            + '<div class="se-kpi-value">' + esc(value) + '</div></div>';
    }

    // ------------------------------------------------------ stream health card --

    function healthCard(health) {
        var problems = [];
        if (health.dropRate > 0.01) {
            problems.push(bi('遺失 ' + (health.dropRate * 100).toFixed(1) + '% 的取樣——請檢查 WiFi 距離。',
                'Lost ' + (health.dropRate * 100).toFixed(1) + '% of samples -- check WiFi range.'));
        }
        if (health.saturatedAccel > 0) {
            problems.push(bi('加速度計在 ' + health.saturatedAccel + ' 個取樣點飽和——揮拍超出量測範圍，'
                + '力量相關數值會被低估。',
                'Accelerometer saturated on ' + health.saturatedAccel + ' samples -- the stroke exceeded '
                + 'full scale, so force figures are under-reported.'));
        }
        if (health.saturatedGyro > 0) {
            problems.push(bi('陀螺儀在 ' + health.saturatedGyro + ' 個取樣點飽和——'
                + '角速度與拍面角度會被低估。',
                'Gyroscope saturated on ' + health.saturatedGyro + ' samples -- rotation rate and '
                + 'face angle are under-reported.'));
        }

        return '<div class="se-card' + (problems.length ? ' se-card-warn' : '') + '">'
            + '<h4>' + bi('📡 感測器串流品質', '📡 Sensor stream health') + '</h4>'
            + '<div class="se-kpis">'
            + kpi(bi('取樣數', 'Samples'), String(health.rows))
            + kpi(bi('實際取樣率', 'Effective rate'), num(health.effectiveHz, 2) + ' Hz')
            + kpi(bi('遺失取樣', 'Dropped'), String(health.dropped))
            + kpi(bi('飽和點', 'Saturated'), (health.saturatedAccel + health.saturatedGyro) + '')
            + '</div>'
            + (problems.length
                ? '<ul class="se-list">' + problems.map(function (p) { return '<li>' + p + '</li>'; }).join('') + '</ul>'
                : '<p class="se-verdict" style="color:#2d6a4f;">'
                + bi('資料串流乾淨——沒有遺失取樣，也沒有量程飽和。',
                    'Clean stream -- no dropped samples, no range saturation.') + '</p>')
            + '</div>';
    }

    // --------------------------------------------------------- per-swing table --

    function swingTable(analyzed) {
        if (!analyzed || !analyzed.length) {
            return '<div class="se-card"><p>' + bi('未偵測到任何擊球。', 'No strokes detected.') + '</p></div>';
        }

        var rows = analyzed.map(function (s, i) {
            var f = s.face;
            var faceCell = (f && f.trustworthy)
                ? num(f.closeOpenDeg, 0) + '° <span class="se-pm">±' + num(f.uncertaintyDeg, 0) + '°</span>'
                : '<span class="se-dim">' + (f ? '可信度不足 / low confidence' : '—') + '</span>';
            var impactCell = (s.impact && s.impact.applicable)
                ? num(s.impact.ringRmsG, 2) + ' g'
                : '<span class="se-dim">' + esc('空揮 / shadow') + '</span>';
            return '<tr>'
                + '<td>#' + (i + 1) + '</td>'
                + '<td>' + num(s.tContact, 2) + ' s</td>'
                + '<td>' + num(s.peakGyroDps, 0) + '</td>'
                + '<td>' + num(s.energy.headSpeedKmh, 1) + '</td>'
                + '<td>' + num(s.peakAccG, 2) + '</td>'
                + '<td>' + faceCell + '</td>'
                + '<td>' + num(s.tempo.backswingMs, 0) + ' / ' + num(s.tempo.followThroughMs, 0) + '</td>'
                + '<td>' + impactCell + '</td>'
                + '</tr>';
        }).join('');

        return '<div class="se-card">'
            + '<h4>' + bi('🎾 逐球細節', '🎾 Per-stroke detail') + '</h4>'
            + '<div class="se-scroll"><table class="se-table"><thead><tr>'
            + '<th>' + bi('球', 'Stroke') + '</th>'
            + '<th>' + bi('擊球時間', 'Contact') + '</th>'
            + '<th>' + bi('角速度 (°/s)', 'Rotation (°/s)') + '</th>'
            + '<th>' + bi('拍頭 (km/h)*', 'Head (km/h)*') + '</th>'
            + '<th>' + bi('加速度 (g)', 'Accel (g)') + '</th>'
            + '<th>' + bi('拍面轉動', 'Face rotation') + '</th>'
            + '<th>' + bi('引拍 / 隨球 (ms)', 'Back / follow (ms)') + '</th>'
            + '<th>' + bi('擊球震動†', 'Strike ring†') + '</th>'
            + '</tr></thead><tbody>' + rows + '</tbody></table></div>'
            + '<p class="se-note">' + bi(
                '* 拍頭速度是以角速度乘上假設的揮拍半徑估算（預設 1.0 m）。'
                + '絕對數值取決於這個半徑；但球與球之間的比較一律有效。'
                + '拍面角度是相對於你自己準備姿勢的轉動量，而非相對於球場的絕對角度。'
                + ' † 擊球震動只有在實際擊球時才有意義；空揮沒有撞擊可測，因此標示為「空揮」。',
                '* Head speed is estimated as angular rate times an assumed swing radius (default 1.0 m). '
                + 'Its absolute value depends on that radius; comparisons between strokes are always valid. '
                + 'Face angle is rotation relative to your own ready position, not an absolute angle to the court.'
                + ' † Strike ring only means something on a real ball strike; a shadow swing has no impact to '
                + 'characterise, so it reads "shadow".')
            + '</p></div>';
    }

    // ------------------------------------------------------------ what camera --

    /**
     * The panel that answers "what did the sensor add?". Deliberately explicit:
     * the point of the sensor mode is the measurements a single frontal camera
     * physically cannot make, and the UI should name them rather than leave the
     * user to infer that the two modes differ.
     */
    function sensorAddedCard(analyzed, sync) {
        var withFace = analyzed.filter(function (s) { return s.face && s.face.trustworthy; }).length;
        return '<div class="se-card se-card-accent">'
            + '<h4>' + bi('➕ 感測器比單靠攝影機多提供了什麼',
                '➕ What the sensor adds over camera alone') + '</h4>'
            + '<ul class="se-list">'
            + '<li>' + bi(
                '擊球瞬間的拍面轉動量——' + analyzed.length + ' 球中測得 ' + withFace
                + ' 球。單一正面 2D 攝影機完全看不到這個量。',
                'Racket-face rotation at contact -- measured on ' + withFace + '/' + analyzed.length
                + ' strokes. A single frontal 2D camera cannot see this quantity at all.') + '</li>'
            + '<li>' + bi(
                '擊球時間解析度 5 ms（200 Hz），而非攝影機的 '
                + (sync && sync.frameMs ? num(sync.frameMs, 0) : '33') + ' ms——'
                + '揮拍節奏的量測精細約 6 倍。',
                'Contact timing at 5 ms resolution (200 Hz) instead of the camera\'s '
                + (sync && sync.frameMs ? num(sync.frameMs, 0) : '33') + ' ms -- '
                + 'about 6x finer for swing tempo.') + '</li>'
            + '<li>' + bi(
                '球拍真實的角速度與加速度，而不是從影像中手腕像素間接推算。',
                'True racket angular rate and acceleration, instead of inferring them from wrist pixels.') + '</li>'
            + '<li>' + bi(
                '由感測器定位的擊球瞬間可重新校準攝影機自行推估的擊球影格——'
                + '連帶提升骨架分析的準確度。',
                'The sensor-located contact instant re-anchors the camera\'s own contact frame -- '
                + 'improving the skeleton analysis as well.') + '</li>'
            + '</ul></div>';
    }

    // ------------------------------------------------------------- session --

    function consistencyCard(cons) {
        var color = LEVEL_COLORS[cons.band.level] || '#888';
        var rows = cons.perMetric.map(function (m) {
            if (m.n < 2) {
                return '<tr><td>' + bi(m.zh, m.en) + '</td><td colspan="4" class="se-dim">'
                    + bi('資料不足', 'not enough data') + '</td></tr>';
            }
            var c = LEVEL_COLORS[m.band.level] || '#888';
            return '<tr>'
                + '<td>' + bi(m.zh, m.en) + '</td>'
                + '<td>' + num(m.mean, 1) + ' ' + esc(m.unit) + '</td>'
                + '<td>' + num(m.sd, 1) + '</td>'
                + '<td><strong style="color:' + c + ';">' + num(m.cv, 1) + '%</strong></td>'
                + '<td style="color:' + c + ';">' + bi(m.band.zh, m.band.en) + '</td>'
                + '</tr>';
        }).join('');

        return '<div class="se-card" style="border-left-color:' + color + ';">'
            + '<h4>' + bi('📈 本次練習穩定度(至少3次)', '📈 Session consistency(at least 3 times)') + '</h4>'
            + '<div class="se-kpis">'
            + kpi(bi('穩定度分數', 'Stability score'), (cons.score == null ? '—' : cons.score + ' / 100'))
            + kpi(bi('綜合 CV', 'Headline CV'), num(cons.headlineCv, 1) + '%')
            + kpi(bi('球數', 'Strokes'), String(cons.swingCount))
            + '</div>'
            + '<p class="se-verdict" style="color:' + color + ';">' + bi(cons.band.zh, cons.band.en) + '</p>'
            + '<div class="se-scroll"><table class="se-table"><thead><tr>'
            + '<th>' + bi('指標', 'Metric') + '</th><th>' + bi('平均', 'Mean') + '</th>'
            + '<th>' + bi('標準差', 'SD') + '</th><th>CV</th><th>' + bi('評級', 'Band') + '</th>'
            + '</tr></thead><tbody>' + rows + '</tbody></table></div>'
            + '<p class="se-note">' + bi(
                'CV（變異係數）＝標準差 ÷ 平均值，越低代表重現性越好。'
                + '此處的評級門檻是為了易讀而選定的顯示閾值，'
                + '尚未是針對本硬體校準過的標準。',
                'CV (coefficient of variation) = standard deviation / mean. Lower is more repeatable. '
                + 'The bands here are readability thresholds chosen by the team, '
                + 'not norms calibrated for this hardware.') + '</p>'
            + '</div>';
    }

    function fatigueCard(f) {
        if (!f.ok) {
            return '<div class="se-card"><h4>' + bi('😮‍💨 疲勞趨勢', '😮‍💨 Fatigue trend') + '</h4>'
                + '<p class="se-dim">' + bi(f.zh, f.en) + '</p></div>';
        }
        var color = f.worsened ? LEVEL_COLORS.fair : LEVEL_COLORS.good;
        return '<div class="se-card" style="border-left-color:' + color + ';">'
            + '<h4>' + bi('😮‍💨 疲勞趨勢', '😮‍💨 Fatigue trend') + '</h4>'
            + '<div class="se-kpis">'
            + kpi(bi('前半段 CV', 'First half CV'), num(f.firstHalfCv, 1) + '%')
            + kpi(bi('後半段 CV', 'Second half CV'), num(f.secondHalfCv, 1) + '%')
            + '</div>'
            + '<p class="se-verdict" style="color:' + color + ';">' + bi(f.zh, f.en) + '</p></div>';
    }

    function bestPairCard(pair, coaching) {
        if (!pair || !pair.ok) {
            return '<div class="se-card"><h4>' + bi('⭐ 最好的兩球', '⭐ Best two strokes') + '</h4>'
                + '<p class="se-dim">' + bi('至少需要 2 球。', 'Need at least 2 strokes.') + '</p></div>';
        }

        var head = '<div class="se-card se-card-accent">'
            + '<h4>' + bi('⭐ 以最好的兩球作為參考基準',
                '⭐ Best two strokes as your reference') + '</h4>'
            + '<div class="se-kpis">'
            + kpi(bi('選定的一對', 'Selected pair'), '#' + (pair.indices[0] + 1) + ' & #' + (pair.indices[1] + 1))
            + kpi(bi('相隔', 'Apart by'), pair.indexGap + bi(' 球', ' strokes'))
            + kpi(bi('力道差異', 'Effort gap'), num(pair.metricGapPct, 1) + '%')
            + kpi(bi('拍面角差', 'Face gap'), num(pair.faceSpreadDeg, 1) + '°')
            + '</div>';

        if (!coaching.ok) {
            return head + '<p class="se-verdict" style="color:' + LEVEL_COLORS.fair + ';">'
                + coaching.zh + '</p></div>';
        }

        var items = coaching.items.map(function (it) {
            var cls = !it.measurable ? 'se-dim' : (it.actionable ? 'se-action' : 'se-ok');
            return '<li class="' + cls + '">' + it.zh + '</li>';
        }).join('');

        return head
            + '<p class="se-verdict">' + coaching.zh + '</p>'
            + '<ul class="se-list">' + items + '</ul>'
            + '<p class="se-note">' + '只有當偏差超過量測誤差（±'
                + num(coaching.toleranceDeg, 0) + '°）時才會提出建議。參考基準取自你自己最好的兩球，'
                + '而不是教科書上的標準角度——未校準安裝方位的六軸 IMU 只能量測'
                + '相對於準備姿勢的轉動量，無法量測相對於球場的絕對角度。' + '</p>'
            + '</div>';
    }

    /**
     * Second opinion on the stroke label, from the racket's rotation direction.
     * Rendered only when it actually has something to say.
     */
    function directionCard(dc) {
        if (!dc || !dc.ok) return '';
        var color = dc.consistent ? LEVEL_COLORS.good : LEVEL_COLORS.fair;
        return '<div class="se-card" style="border-left-color:' + color + ';">'
            + '<h4>' + bi('🔍 擊球類型交叉驗證', '🔍 Stroke-type cross-check') + '</h4>'
            + '<p class="se-verdict" style="color:' + color + ';">' + bi(dc.zh, dc.en) + '</p>'
            + '<p class="se-note">' + bi(
                '依據球拍繞自身長軸的轉動方向，正手與反手方向相反。這不是分類器——'
                + '轉動的正負號取決於感測器安裝方位，所以只比較「同一批標記相同的球」之間是否一致，'
                + '不會單獨判定某一球是正手還是反手。',
                'Based on which way the racket turns about its own long axis; forehands and backhands turn '
                + 'opposite ways. This is not a classifier -- the sign depends on how the sensor is mounted, '
                + 'so it only compares strokes within one same-labelled group rather than naming any stroke '
                + 'on its own.') + '</p>'
            + '</div>';
    }

    // ------------------------------------------------------------ stylesheet --

    var CSS = [
        '.se-card{background:#fff;border-radius:16px;padding:18px 20px;margin-bottom:16px;',
        '  border-left:4px solid #2d6a4f;box-shadow:0 4px 16px rgba(0,0,0,0.05);}',
        '.se-card h4{font-size:0.98rem;color:#2d6a4f;margin-bottom:12px;font-weight:700;}',
        '.se-card-warn{border-left-color:#d97706;background:#fffbeb;}',
        '.se-card-accent{background:#f0fdf4;}',
        '.se-kpis{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:12px;}',
        '.se-kpi{flex:1 1 110px;background:#f7faf8;border:1px solid #e3efe7;border-radius:12px;padding:10px 12px;}',
        '.se-kpi-label{font-size:0.7rem;color:#667;font-weight:600;line-height:1.3;}',
        '.se-kpi-value{font-size:1.12rem;font-weight:800;color:#2d6a4f;margin-top:3px;}',
        '.se-verdict{font-size:0.88rem;font-weight:600;margin:10px 0;}',
        '.se-note{font-size:0.76rem;color:#6b7280;line-height:1.55;margin-top:10px;}',
        '.se-list{list-style:none;padding:0;margin:8px 0;}',
        '.se-list li{font-size:0.83rem;color:#374151;padding:6px 0 6px 20px;position:relative;line-height:1.5;}',
        '.se-list li::before{content:"•";position:absolute;left:4px;color:#2d6a4f;font-weight:700;}',
        '.se-list li.se-action::before{content:"▸";color:#d97706;}',
        '.se-list li.se-ok::before{content:"✓";color:#2d6a4f;}',
        '.se-list li.se-dim{color:#9ca3af;}',
        '.se-list li.se-dim::before{content:"–";color:#9ca3af;}',
        '.se-scroll{overflow-x:auto;-webkit-overflow-scrolling:touch;}',
        '.se-table{width:100%;border-collapse:collapse;font-size:0.8rem;min-width:520px;}',
        '.se-table th{background:#f7faf8;color:#2d6a4f;font-weight:700;text-align:left;padding:8px 10px;',
        '  border-bottom:2px solid #e3efe7;white-space:nowrap;}',
        '.se-table td{padding:8px 10px;border-bottom:1px solid #eef4f0;}',
        '.se-table tr:hover td{background:#fafcfb;}',
        '.se-pm{color:#9ca3af;font-size:0.72rem;}',
        '.se-dim{color:#9ca3af;}',
        '.se-progress{height:6px;background:#e8f5e9;border-radius:99px;overflow:hidden;margin:8px 0;}',
        '.se-progress > div{height:100%;background:#2d6a4f;transition:width 0.2s;}'
    ].join('\n');

    function injectStyles(doc) {
        var d = doc || document;
        if (d.getElementById('skilleye-report-css')) return;
        var el = d.createElement('style');
        el.id = 'skilleye-report-css';
        el.textContent = CSS;
        d.head.appendChild(el);
    }

    return {
        esc: esc, bi: bi, num: num, kpi: kpi,
        syncCard: syncCard,
        videoIssueCard: videoIssueCard,
        healthCard: healthCard,
        swingTable: swingTable,
        sensorAddedCard: sensorAddedCard,
        consistencyCard: consistencyCard,
        fatigueCard: fatigueCard,
        bestPairCard: bestPairCard,
        directionCard: directionCard,
        injectStyles: injectStyles,
        CSS: CSS
    };
}));
