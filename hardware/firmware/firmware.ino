// skilleye rev2.x streaming firmware -- XIAO ESP32-C6 + GY-521 (MPU6050)
//
// Mounted at the racket throat. The ESP32 is the *host*: it brings up its own
// WiFi access point, the laptop joins that network, and the moment the laptop
// opens a TCP connection the IMU starts streaming CSV rows at 200 Hz.
//
//   1. Flash this sketch (Arduino IDE: Seeed XIAO_ESP32C6, "USB CDC On Boot"
//      = Enabled). Serial Monitor at 115200 prints the AP details.
//      Needs esp32 core 3.x -- uses WiFiServer::accept(), not the old
//      deprecated available(). Built against 3.3.10.
//   2. Copy secrets.example.h to secrets.h, then join the WiFi network named
//      there (SSID and password live in secrets.h, which is gitignored).
//   3. Run:  cd hardware/client && python imu_client.py monitor
//      (or  python imu_client.py record --out take01.csv)
//
// Wire format, sent as soon as a client connects:
//
//   # skilleye imu stream v1
//   # rate_hz=200 accel_range_g=16 gyro_range_dps=2000 ...      <- metadata
//   seq,t_us,ax,ay,az,gx,gy,gz                                  <- column header
//   0,1234567,0.0122,-0.9976,0.0341,1.22,-0.61,0.00             <- rows
//
// Column order is [ax, ay, az, gx, gy, gz], the standard MPU6050 register
// order, so the CSV needs no remapping wherever it ends up.
//
// If the IMU doesn't come up, this still runs the old bring-up sketch's I2C bus
// scan on the failure path and prints what it finds.
//
// Wiring per hardware/skilleye_2.0/skilleye_2.0.kicad_sch:
//   SDA     -> D4  (default Wire SDA, no remap needed)
//   SCL     -> D5  (default Wire SCL)
//   IMU_INT -> D0  (wired but unused -- we clock sampling off the ESP32 timer)

#include <Wire.h>
#include <WiFi.h>
#include <esp_timer.h>

// AP_SSID / AP_PASSWORD / TCP_PORT live in secrets.h, which is gitignored so
// the device's credentials stay out of the repo. __has_include keeps a fresh
// clone from dying on a missing-file error and points at the fix instead.
#if __has_include("secrets.h")
#include "secrets.h"
#else
#error "Missing secrets.h -- copy hardware/firmware/secrets.example.h to secrets.h and set your AP password."
#endif

// ---------------------------------------------------------------- config ----

#define AP_CHANNEL       1

#define SAMPLE_RATE_HZ   200             // 100-200 Hz resolves the contact spike
#define RING_SLOTS       512             // 512 / 200 Hz = 2.5 s of slack

// The XIAO ESP32-C6 has two antennas behind an RF switch, and it comes up on
// the *wrong* one unless firmware says otherwise -- symptom is a WiFi AP that
// works at 30 cm and drops out across the room. Set to 0 when building for the
// bare ESP32-C6-WROOM-1 on the rev2.x PCB, which has no such switch.
#define BOARD_XIAO_ESP32C6   1
#define USE_EXTERNAL_ANTENNA 0           // 1 only if a u.FL antenna is attached

// The rev2.1 board has no battery divider routed to a GPIO, so this is off.
// If you bodge one on (VBAT -> 2x 100k -> GND, tap to an ADC pin), set this to
// that pin (e.g. D1) and the stream header will report volts.
#define BATT_SENSE_PIN   -1
#define BATT_DIVIDER     2.0f            // (R_top + R_bottom) / R_bottom

// ---- MPU6050 registers ----
static const uint8_t REG_SMPLRT_DIV   = 0x19;
static const uint8_t REG_CONFIG       = 0x1A;
static const uint8_t REG_GYRO_CONFIG  = 0x1B;
static const uint8_t REG_ACCEL_CONFIG = 0x1C;
static const uint8_t REG_ACCEL_XOUT_H = 0x3B;
static const uint8_t REG_PWR_MGMT_1   = 0x6B;
static const uint8_t REG_WHO_AM_I     = 0x75;

// GY-521 leaves AD0 pulled low -> 0x68. Jumpered high it becomes 0x69, so we
// probe both and use whichever answers.
static const uint8_t ADDR_LOW  = 0x68;
static const uint8_t ADDR_HIGH = 0x69;

// Full-scale ranges, deliberately wide. The bring-up sketch used the +/-2 g,
// +/-250 deg/s defaults, which are fine for "is it alive" but clip badly on a
// real stroke: ball contact at the racket throat swings well past 2 g, and a
// serve's racket rotation passes 250 deg/s long before contact. A clipped
// contact spike is exactly the feature the fusion model is meant to see, so we
// trade resolution for headroom and run at the sensor's widest ranges.
static const uint8_t ACCEL_FS_SEL = 0x18;   // +/-16 g
static const uint8_t GYRO_FS_SEL  = 0x18;   // +/-2000 deg/s
static const float ACCEL_LSB_PER_G  = 2048.0f;
static const float GYRO_LSB_PER_DPS = 16.4f;

// DLPF 94 Hz against a 100 Hz Nyquist (200 Hz sampling). Sitting right at the
// edge is intentional -- tightening the filter would round off the contact
// spike, which matters more here than textbook anti-aliasing margin.
static const uint8_t DLPF_CFG = 0x02;
static const int DLPF_HZ = 94;

// ------------------------------------------------------------- ring buffer --

struct Sample {
  uint32_t seq;
  uint64_t t_us;
  int16_t  a[3];
  int16_t  g[3];
};

static Sample ring[RING_SLOTS];
static volatile uint32_t ringHead = 0;      // written by sampler task only
static volatile uint32_t ringTail = 0;      // written by network loop only
static volatile uint32_t seqCounter = 0;
static volatile uint32_t droppedCount = 0;  // samples lost to a full buffer
static volatile bool streaming = false;

static uint8_t imuAddr = 0;
static SemaphoreHandle_t sampleTick = nullptr;
static esp_timer_handle_t sampleTimer = nullptr;

static WiFiServer server(TCP_PORT);
static WiFiClient client;

// ------------------------------------------------------------ I2C helpers ---

static bool writeReg(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(imuAddr);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

static bool readRegs(uint8_t reg, uint8_t *buf, size_t len) {
  Wire.beginTransmission(imuAddr);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;      // repeated start
  // Cast both args to int to land exactly on requestFrom(int, int) -- the
  // ESP32 core's overload set is otherwise ambiguous for a uint8_t address.
  if (Wire.requestFrom((int)imuAddr, (int)len) != (int)len) return false;
  for (size_t i = 0; i < len; i++) buf[i] = Wire.read();
  return true;
}

static bool ping(uint8_t addr) {
  Wire.beginTransmission(addr);
  return Wire.endTransmission() == 0;
}

static void busScan() {
  Serial.println("I2C scan:");
  uint8_t found = 0;
  for (uint8_t a = 0x08; a < 0x78; a++) {
    if (ping(a)) {
      Serial.printf("  device at 0x%02X\n", a);
      found++;
    }
  }
  if (!found) Serial.println("  nothing found -- check SDA/SCL/3V3/GND");
}

static bool imuInit() {
  if      (ping(ADDR_LOW))  imuAddr = ADDR_LOW;
  else if (ping(ADDR_HIGH)) imuAddr = ADDR_HIGH;
  else return false;

  uint8_t who = 0;
  if (!readRegs(REG_WHO_AM_I, &who, 1)) return false;
  // Genuine InvenSense parts return 0x68; clones commonly report 0x70/0x72/0x98.
  Serial.printf("MPU6050 at 0x%02X, WHO_AM_I = 0x%02X%s\n",
                imuAddr, who, who == 0x68 ? " (genuine)" : " (clone, usually fine)");

  if (!writeReg(REG_PWR_MGMT_1, 0x00)) return false;       // wake from sleep
  delay(100);
  // Internal rate stays at 1 kHz while we sample at 200 Hz. Running the sensor
  // faster than we read it means every read returns a genuinely fresh sample --
  // no duplicated rows, and no slow beat artifact from the MPU's oscillator
  // drifting against the ESP32's crystal.
  writeReg(REG_SMPLRT_DIV,   0x00);
  writeReg(REG_CONFIG,       DLPF_CFG);
  writeReg(REG_GYRO_CONFIG,  GYRO_FS_SEL);
  writeReg(REG_ACCEL_CONFIG, ACCEL_FS_SEL);
  return true;
}

// ---------------------------------------------------------------- sampling --

// Dispatched from the esp_timer *task* (ESP_TIMER_TASK), not an interrupt, so
// this is a plain xSemaphoreGive and not the FromISR variant. Kept to just the
// give so the I2C read -- ~380 us at 400 kHz -- happens on our own task rather
// than stalling every other timer in the system, WiFi's included.
static void onSampleTimer(void *) {
  xSemaphoreGive(sampleTick);
}

// The tick is a binary semaphore, so if this task ever gets starved long
// enough to miss a tick the extra give is coalesced rather than queued. That
// costs a sample but never corrupts the timeline: every row carries its own
// t_us, so a missed tick shows up honestly as a sub-200 Hz measured rate on
// the client instead of as evenly-spaced rows that silently lie about timing.
static void samplerTask(void *) {
  for (;;) {
    if (xSemaphoreTake(sampleTick, portMAX_DELAY) != pdTRUE) continue;
    if (!streaming) continue;

    uint8_t raw[14];
    uint64_t t = (uint64_t)esp_timer_get_time();
    if (!readRegs(REG_ACCEL_XOUT_H, raw, sizeof(raw))) continue;

    uint32_t seq = seqCounter++;
    uint32_t head = ringHead;
    uint32_t next = (head + 1) % RING_SLOTS;
    if (next == ringTail) {
      // Consumer fell behind (WiFi stall). Drop rather than block the sampler,
      // and count it -- seq still increments, so the gap is visible in the CSV
      // instead of silently corrupting the timeline of a training clip.
      droppedCount++;
      continue;
    }

    Sample &s = ring[head];
    s.seq  = seq;
    s.t_us = t;
    // Burst layout: accel XYZ, temperature, gyro XYZ -- all big-endian int16.
    s.a[0] = (int16_t)((raw[0]  << 8) | raw[1]);
    s.a[1] = (int16_t)((raw[2]  << 8) | raw[3]);
    s.a[2] = (int16_t)((raw[4]  << 8) | raw[5]);
    s.g[0] = (int16_t)((raw[8]  << 8) | raw[9]);
    s.g[1] = (int16_t)((raw[10] << 8) | raw[11]);
    s.g[2] = (int16_t)((raw[12] << 8) | raw[13]);

    // Publish only after the slot is fully written.
    __sync_synchronize();
    ringHead = next;
  }
}

// ------------------------------------------------------------ TCP plumbing --

// WiFiClient::write can come up short when the socket buffer fills; keep
// pushing until it's all out or the peer goes away.
static bool writeAll(const uint8_t *buf, size_t len) {
  size_t sent = 0;
  while (sent < len) {
    if (!client.connected()) return false;
    size_t n = client.write(buf + sent, len - sent);
    if (n == 0) {
      delay(1);
      continue;
    }
    sent += n;
  }
  return true;
}

static bool writeLine(const char *s) {
  return writeAll((const uint8_t *)s, strlen(s));
}

static void sendHeader() {
  char buf[256];

  writeLine("# skilleye imu stream v1\n");
  snprintf(buf, sizeof(buf), "# device=xiao_esp32c6 imu=mpu6050 addr=0x%02X\n", imuAddr);
  writeLine(buf);
  snprintf(buf, sizeof(buf),
           "# rate_hz=%d accel_range_g=16 gyro_range_dps=2000 dlpf_hz=%d\n",
           SAMPLE_RATE_HZ, DLPF_HZ);
  writeLine(buf);
  // Phrased without '=' on purpose: the client parses `key=value` out of every
  // comment, and a units line written that way pollutes the device metadata.
  writeLine("# units: accel in g, gyro in deg/s, t_us in microseconds since boot (monotonic)\n");
  writeLine("# no shared clock with the camera -- tap the racket once before each take\n");
  writeLine("# so both recordings share a spike to align on\n");

#if BATT_SENSE_PIN >= 0
  float vbat = analogReadMilliVolts(BATT_SENSE_PIN) * BATT_DIVIDER / 1000.0f;
  snprintf(buf, sizeof(buf), "# battery_v=%.2f\n", vbat);
  writeLine(buf);
#endif

  writeLine("seq,t_us,ax,ay,az,gx,gy,gz\n");
}

static void startSession() {
  // Quiesce the producer, reset the buffer, then let it run -- each connection
  // starts at seq 0 so the laptop can detect gaps without knowing uptime.
  streaming = false;
  delay(20);
  ringHead = ringTail = 0;
  seqCounter = 0;
  droppedCount = 0;
  __sync_synchronize();

  sendHeader();
  streaming = true;
}

// Drain whatever the client sends. Nothing is interpreted -- this only keeps
// the RX buffer from filling up and stalling the socket.
static void drainInput() {
  while (client.available()) client.read();
}

// ------------------------------------------------------------------ setup ---

static void setupAntenna() {
#if BOARD_XIAO_ESP32C6
  // WIFI_ENABLE (GPIO3) and WIFI_ANT_CONFIG (GPIO14) come from the
  // XIAO_ESP32C6 variant's pins_arduino.h.
  pinMode(WIFI_ENABLE, OUTPUT);
  digitalWrite(WIFI_ENABLE, LOW);              // power the RF switch
  pinMode(WIFI_ANT_CONFIG, OUTPUT);
  digitalWrite(WIFI_ANT_CONFIG, USE_EXTERNAL_ANTENNA ? HIGH : LOW);
#endif
}

void setup() {
  Serial.begin(115200);
  delay(2000);                                             // let USB CDC enumerate
  Serial.println("\n=== skilleye imu stream ===");

  Wire.begin();                                            // D4 = SDA, D5 = SCL
  Wire.setClock(400000);

  if (!imuInit()) {
    Serial.println("MPU6050 init FAILED -- halting");
    busScan();
    while (true) delay(1000);
  }

  setupAntenna();
  WiFi.mode(WIFI_AP);
  if (!WiFi.softAP(AP_SSID, AP_PASSWORD, AP_CHANNEL, 0, 4)) {
    Serial.println("softAP() FAILED -- halting");
    while (true) delay(1000);
  }
  WiFi.setSleep(false);                                    // keep latency down

  server.begin();
  server.setNoDelay(true);

  Serial.printf("AP  SSID : %s\n", AP_SSID);
  Serial.printf("AP  pass : %s\n", AP_PASSWORD);
  Serial.printf("stream   : %s:%d\n", WiFi.softAPIP().toString().c_str(), TCP_PORT);
  Serial.println("waiting for the laptop to connect...");

#ifdef LED_BUILTIN
  pinMode(LED_BUILTIN, OUTPUT);
#endif

  sampleTick = xSemaphoreCreateBinary();
  if (sampleTick == nullptr) {
    Serial.println("semaphore alloc FAILED -- halting");
    while (true) delay(1000);
  }
  xTaskCreate(samplerTask, "sampler", 4096, nullptr, 5, nullptr);

  const esp_timer_create_args_t args = {
    .callback = &onSampleTimer,
    .arg = nullptr,
    .dispatch_method = ESP_TIMER_TASK,
    .name = "imu_tick",
    .skip_unhandled_events = true,
  };
  ESP_ERROR_CHECK(esp_timer_create(&args, &sampleTimer));
  ESP_ERROR_CHECK(esp_timer_start_periodic(sampleTimer, 1000000ULL / SAMPLE_RATE_HZ));
}

// ------------------------------------------------------------------- loop ---

// Solid while streaming, slow blink while waiting for a client -- the only
// status indicator on a board that's taped to a racket.
static void updateLed(bool active) {
#ifdef LED_BUILTIN
  bool on = active ? true : ((millis() / 500) % 2 == 0);
  digitalWrite(LED_BUILTIN, on ? LOW : HIGH);   // XIAO user LED is active-low
#endif
}

static void acceptClient() {
  WiFiClient incoming = server.accept();     // available() is deprecated in core 3.x
  if (!incoming) return;

  if (client && client.connected()) {
    // One recording at a time -- a second stream would interleave rows.
    incoming.println("# busy: another client is already streaming");
    incoming.stop();
    return;
  }

  client = incoming;
  client.setNoDelay(true);
  Serial.printf("client connected from %s\n", client.remoteIP().toString().c_str());
  startSession();
}

void loop() {
  if (!client || !client.connected()) {
    if (streaming) {
      streaming = false;
      Serial.println("client disconnected -- waiting for the laptop again");
      client.stop();
    }
    updateLed(false);
    acceptClient();
    delay(2);
    return;
  }

  updateLed(true);
  drainInput();

  // Report drops as a comment row so a gap in seq has a visible cause in the
  // recorded file rather than looking like a sensor fault later.
  uint32_t drops = droppedCount;
  static uint32_t reportedDrops = 0;
  if (drops != reportedDrops) {
    char note[64];
    snprintf(note, sizeof(note), "# dropped=%u\n", (unsigned)drops);
    writeLine(note);
    reportedDrops = drops;
  }

  // Batch rows into one buffer so a 200 Hz stream isn't 200 tiny TCP writes.
  char out[1400];
  size_t used = 0;
  while (ringTail != ringHead) {
    const Sample &s = ring[ringTail];
    int n = snprintf(out + used, sizeof(out) - used,
                     "%u,%llu,%.4f,%.4f,%.4f,%.2f,%.2f,%.2f\n",
                     (unsigned)s.seq, (unsigned long long)s.t_us,
                     s.a[0] / ACCEL_LSB_PER_G,
                     s.a[1] / ACCEL_LSB_PER_G,
                     s.a[2] / ACCEL_LSB_PER_G,
                     s.g[0] / GYRO_LSB_PER_DPS,
                     s.g[1] / GYRO_LSB_PER_DPS,
                     s.g[2] / GYRO_LSB_PER_DPS);
    if (n < 0) break;
    if ((size_t)n >= sizeof(out) - used) break;   // no room; flush and come back
    used += n;
    ringTail = (ringTail + 1) % RING_SLOTS;
    if (sizeof(out) - used < 128) break;          // headroom for one more row
  }

  if (used && !writeAll((const uint8_t *)out, used)) {
    Serial.println("write failed -- dropping client");
    streaming = false;
    client.stop();
    return;
  }

  if (!used) delay(1);                            // buffer empty, yield
}
