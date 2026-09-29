"""Almacén de LSM Coach: SQLite local + sincronización con Supabase.

Todo se guarda primero en local, así la app nunca espera a internet. Un hilo sube
lo pendiente a Supabase cuando hay conexión (tablas: db/supabase_schema.sql).

Configuración en .env (está en .gitignore; NUNCA subirlo ni ponerlo en la página web):
    SUPABASE_URL=https://xxxx.supabase.co
    SUPABASE_KEY=<clave secreta: "secret" / "service_role">

Uso por consola:
    python store.py check     # prueba conexión, clave y tablas (no escribe nada)
    python store.py status    # cuántas filas hay y cuántas faltan por subir
    python store.py sync      # sube lo pendiente ahora
"""
import json
import os
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone

DB_FILE = "lsm_coach.db"
ENV_FILE = ".env"
SYNC_EVERY_S = 30
BATCH = 500

# Columnas que se suben a Supabase (en ese orden); las JSON se guardan como texto en SQLite
COLUMNS = {
    "samples": ["id", "created_at", "person", "sign", "is_error", "source", "camera",
                "angles", "landmarks", "imu"],
    "attempts": ["id", "created_at", "person", "sign", "seconds", "failed_parameters"],
}
JSON_COLUMNS = {"angles", "landmarks", "imu", "failed_parameters"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, person TEXT NOT NULL, sign TEXT NOT NULL,
    is_error INTEGER NOT NULL DEFAULT 0, source TEXT NOT NULL DEFAULT 'propio', camera TEXT,
    angles TEXT NOT NULL, landmarks TEXT, imu TEXT, synced INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS attempts (
    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, person TEXT NOT NULL, sign TEXT NOT NULL,
    seconds REAL, failed_parameters TEXT, synced INTEGER NOT NULL DEFAULT 0);
"""


def load_env(path=ENV_FILE):
    """Lee KEY=VALUE de .env sin pisar variables que ya existan en el entorno."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class Store:
    def __init__(self, path=DB_FILE, env_file=ENV_FILE):
        if env_file:
            load_env(env_file)
        self.url = os.environ.get("SUPABASE_URL", "").rstrip("/")
        self.key = os.environ.get("SUPABASE_KEY", "")
        self.last_error = None
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.executescript(SCHEMA)

    @property
    def cloud_enabled(self):
        return bool(self.url and self.key)

    def add_sample(self, person, sign, angles, landmarks, imu, is_error=False,
                   source="propio", camera=None):
        return self._insert("samples", {
            "person": person, "sign": sign, "is_error": int(is_error), "source": source,
            "camera": camera, "angles": angles, "landmarks": landmarks, "imu": imu})

    def add_attempt(self, person, sign, seconds, failed_parameters):
        return self._insert("attempts", {"person": person, "sign": sign, "seconds": seconds,
                                         "failed_parameters": failed_parameters})

    def _insert(self, table, row):
        row = {"id": str(uuid.uuid4()),
               "created_at": datetime.now(timezone.utc).isoformat(), **row}
        values = [json.dumps(row[c]) if c in JSON_COLUMNS and row[c] is not None else row[c]
                  for c in COLUMNS[table]]
        with self._lock:
            self._db.execute(f"INSERT INTO {table} ({', '.join(COLUMNS[table])}) "
                             f"VALUES ({', '.join('?' * len(values))})", values)
            self._db.commit()
        return row["id"]

    def count_samples(self, sign, is_error):
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM samples WHERE sign = ? AND is_error = ?",
                                    (sign, int(is_error))).fetchone()[0]

    def pending(self):
        with self._lock:
            return sum(self._db.execute(f"SELECT COUNT(*) FROM {t} WHERE synced = 0").fetchone()[0]
                       for t in COLUMNS)

    def status(self):
        return {"cloud": self.cloud_enabled, "pending": self.pending(), "error": self.last_error}

    # --- sincronización con Supabase (API REST, solo biblioteca estándar) ---

    def _headers(self):
        headers = {"apikey": self.key, "Content-Type": "application/json"}
        if self.key.startswith("eyJ"):  # clave vieja (JWT): también va en Authorization
            headers["Authorization"] = f"Bearer {self.key}"
        return headers

    def _post(self, table, rows):
        headers = {**self._headers(), "Prefer": "resolution=merge-duplicates,return=minimal"}
        req = urllib.request.Request(f"{self.url}/rest/v1/{table}", method="POST",
                                     data=json.dumps(rows).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=15):
            pass

    def check(self):
        """Prueba URL, clave y permisos leyendo 1 fila de cada tabla (no escribe nada).
        Devuelve None si todo está bien, o el problema encontrado."""
        for table in COLUMNS:
            req = urllib.request.Request(f"{self.url}/rest/v1/{table}?select=id&limit=1",
                                         headers=self._headers())
            try:
                with urllib.request.urlopen(req, timeout=15):
                    pass
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")
                if e.code == 401:
                    return f"clave rechazada (401): revisa SUPABASE_KEY. {body[:150]}"
                if "42P01" in body or "PGRST205" in body:
                    return f"no existe la tabla '{table}': ejecuta db/supabase_schema.sql"
                if "42501" in body:
                    return f"sin permisos sobre '{table}': ejecuta otra vez db/supabase_schema.sql"
                return f"{table}: HTTP {e.code} {body[:150]}"
            except (urllib.error.URLError, OSError) as e:
                return f"no se pudo conectar a {self.url}: {e}"
        return None

    def sync_once(self):
        """Sube lo pendiente. Devuelve cuántas filas subió; si falla, lo deja para después."""
        if not self.cloud_enabled:
            return 0
        uploaded = 0
        try:
            for table, cols in COLUMNS.items():
                while True:
                    with self._lock:
                        rows = self._db.execute(f"SELECT {', '.join(cols)} FROM {table} "
                                                f"WHERE synced = 0 LIMIT {BATCH}").fetchall()
                    if not rows:
                        break
                    payload = []
                    for r in rows:
                        d = dict(zip(cols, r))
                        for c in JSON_COLUMNS & d.keys():
                            d[c] = json.loads(d[c]) if d[c] is not None else None
                        if "is_error" in d:
                            d["is_error"] = bool(d["is_error"])
                        payload.append(d)
                    self._post(table, payload)
                    with self._lock:
                        self._db.executemany(f"UPDATE {table} SET synced = 1 WHERE id = ?",
                                             [(d["id"],) for d in payload])
                        self._db.commit()
                    uploaded += len(payload)
            self.last_error = None
        except urllib.error.HTTPError as e:
            self.last_error = f"HTTP {e.code}: {e.read().decode(errors='replace')[:200]}"
        except (urllib.error.URLError, OSError) as e:
            self.last_error = f"sin conexión ({e})"
        return uploaded

    def start_sync(self, every=SYNC_EVERY_S):
        def loop():
            while True:
                self.sync_once()
                time.sleep(every)
        threading.Thread(target=loop, daemon=True).start()


def main():
    store = Store()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd in ("sync", "check") and not store.cloud_enabled:
        sys.exit(f"Falta SUPABASE_URL / SUPABASE_KEY en {ENV_FILE}.")
    if cmd == "check":
        problem = store.check()
        sys.exit(f"ERROR: {problem}" if problem else "OK: conexión, clave y tablas correctas.")
    if cmd == "sync":
        n = store.sync_once()
        print(f"Subidas: {n}" + (f"  Error: {store.last_error}" if store.last_error else ""))
    with store._lock:
        for table in COLUMNS:
            total, pend = store._db.execute(
                f"SELECT COUNT(*), COALESCE(SUM(synced = 0), 0) FROM {table}").fetchone()
            print(f"{table:9s} total={total}  pendientes={pend}")
    print("Nube:", "configurada" if store.cloud_enabled else f"sin configurar (ver {ENV_FILE})")


if __name__ == "__main__":
    main()
