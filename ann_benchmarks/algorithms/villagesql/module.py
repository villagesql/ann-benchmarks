import getpass
import glob
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from multiprocessing.pool import Pool
from itertools import chain
import inspect
import numpy

import pymysql

from ..base.module import BaseANN

# VillageSQL ann-benchmarks adapter.
#
# Adapted from the MariaDB adapter (ann_benchmarks/algorithms/mariadb) to drive
# the vsql_vector extension (SVECTOR type + HNSW custom index) instead of
# MariaDB's built-in VECTOR type. The shape mirrors the MariaDB module so the
# two can be compared apples-to-apples under the same ann-benchmarks harness
# and plot.py, but the SQL surface differs in several ways:
#
#   * Connection uses PyMySQL (VillageSQL speaks the MySQL protocol), not the
#     MariaDB connector.
#   * The vector column is SVECTOR(N) (a vsql_vector custom type) on an InnoDB
#     table; vectors are passed as the text literal '[f0, f1, ...]', not the
#     raw float32 bytes MariaDB's VECTOR type takes.
#   * The index is a SEPARATE statement, not inline in CREATE TABLE:
#         CREATE INDEX idx_v ON t (v hnsw_l2) USING EXTENDED(hnsw)
#                WITH (M = <m>, ef_construction = <efc>)
#     The per-metric index modifier (hnsw_l2 / hnsw_cosine / ...) and the
#     query-side distance function (L2_DISTANCE / COSINE_DISTANCE / ...) must
#     agree.
#   * Query-time search breadth is SET vsql_vector.ef_search = N (a session
#     variable registered by the extension), not mhnsw_ef_search.
#   * Two server-side gates are required for the custom KNN path: preview
#     extensions must be allowed (to INSTALL the preview extension) and the
#     hypergraph optimizer must be on (the classic optimizer will not select
#     the custom index scan).

# Metric -> (index modifier used at build time, distance function used at query
# time, sort order). The modifier and the distance function must name the same
# metric or the index is not used.
_METRICS = {
    "euclidean": {"modifier": "hnsw_l2", "dist_fn": "L2_DISTANCE", "order": "ASC"},
    "cosine": {"modifier": "hnsw_cosine", "dist_fn": "COSINE_DISTANCE", "order": "ASC"},
}


def vector_to_literal(v):
    # SVECTOR accepts a bracketed text literal; a plain repr of the float list
    # is enough. Keep full float32 precision.
    return "[" + ",".join(repr(float(x)) for x in v) + "]"


def many_inserts(arg):
    socket_file, base, embeddings = arg
    conn = pymysql.connect(unix_socket=socket_file, user="root")
    cur = conn.cursor()
    cur.execute("USE ann")
    lenX = len(embeddings)
    start_time = time.time()
    rps = 100
    for i, embedding in enumerate(embeddings):
        while True:
            try:
                cur.execute("INSERT INTO t1 (id, v) VALUES (%s, %s)",
                            (i + base, vector_to_literal(embedding)))
                break
            except pymysql.OperationalError:
                time.sleep(0.01 * (11 + base * 17 % 13))
        if base == 0 and (i + 1) % int(rps + 1) == 0:
            rps = i / (time.time() - start_time)
            print(f"{i:6d} of {lenX}, {rps:4.2f} stmt/sec, ETA {(lenX - i) / rps:.0f} sec")
    cur.execute("commit")


def many_queries(arg):
    socket_file, ef_search, dist_fn, order, n, queries = arg
    conn = pymysql.connect(unix_socket=socket_file, user="root")
    cur = conn.cursor()
    cur.execute("USE ann")
    cur.execute("SET vsql_vector.ef_search = %s" % ef_search)
    res = []
    for v in queries:
        cur.execute(
            f"SELECT id FROM t1 ORDER BY {dist_fn}(v, '{vector_to_literal(v)}') {order} LIMIT {n}")
        res.append([row[0] for row in cur.fetchall()])
    return res


class VillageSQL(BaseANN):

    def __init__(self, metric, method_param):
        self._test_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self._cur = None
        self._perf_proc = None
        self._perf_records = []
        self._perf_stats = []
        self._batch = inspect.stack()[2].frame.f_locals['batch']  # Yo-ho-ho!
        self._m = method_param['M']
        self._ef_construction = method_param.get('ef_construction', 64)
        self._size = 0

        if metric == "angular":
            self._metric = "cosine"
        elif metric == "euclidean":
            self._metric = "euclidean"
        else:
            raise RuntimeError(f"unknown metric {metric}")
        self._modifier = _METRICS[self._metric]["modifier"]
        self._dist_fn = _METRICS[self._metric]["dist_fn"]
        self._order = _METRICS[self._metric]["order"]

        self.prepare_options()
        self.initialize_db()
        self.start_db()

        conn = pymysql.connect(unix_socket=self._socket_file, user="root")
        self._cur = conn.cursor()

    def prepare_options(self):
        self._perf_stat = os.environ.get('PERF', 'no') == 'yes' and VillageSQL.can_run_perf()
        self._perf_record = os.environ.get('FLAMEGRAPH', 'no') == 'yes' and VillageSQL.can_run_flamegraph()
        if self._perf_stat and self._perf_record:
            self._perf_stat = False
            print("\nWarning: Better not to enable both PERF and FLAMEGRAPH. Generating a flame graph only.\n")

        # VillageSQL build dir (the dir with runtime_output_directory/mysqld and
        # veb_output_directory/). INSTALL EXTENSION vsql_vector resolves the VEB
        # by name from veb_output_directory, so the VEB must be there.
        root_dir = os.environ.get('VILLAGESQL_ROOT_DIR')
        if root_dir is None:
            raise RuntimeError(
                "VILLAGESQL_ROOT_DIR is not set. Point it at your VillageSQL build "
                "dir (contains runtime_output_directory/mysqld and "
                "veb_output_directory/vsql_vector.veb).")
        self._mysqld = glob.glob(f"{root_dir}/runtime_output_directory/mysqld")[0]
        self._basedir = root_dir

        # Which VEB gets benchmarked. If VSQL_VECTOR_VEB points at a freshly
        # built vsql_vector.veb, copy it into the server's veb_output_directory
        # here so the run always tests exactly that build — otherwise INSTALL
        # would silently use whatever (possibly stale) copy is already in the
        # server tree. If unset, rely on whatever is already installed there.
        veb_src = os.environ.get('VSQL_VECTOR_VEB')
        if veb_src is not None:
            if not os.path.isfile(veb_src):
                raise RuntimeError(f"VSQL_VECTOR_VEB does not exist: {veb_src}")
            veb_dir = f"{root_dir}/veb_output_directory"
            os.makedirs(veb_dir, exist_ok=True)
            dest = os.path.join(veb_dir, "vsql_vector.veb")
            shutil.copyfile(veb_src, dest)
            print(f"Copied VEB {veb_src} -> {dest}")

        # Initialize the datadir when running locally (vs a prebuilt image).
        self._do_init = os.environ.get('DO_INIT_VILLAGESQL', 'yes') == 'yes'

        workspace = os.environ.get('VILLAGESQL_DB_WORKSPACE')
        if workspace is None:
            raise RuntimeError("Please set VILLAGESQL_DB_WORKSPACE (the scratch database directory).")
        # initialize_db() wipes the datadir before --initialize-insecure (which
        # refuses a non-empty dir). Guard against a mistyped env var pointing at
        # something precious: reject filesystem root / a home directory outright,
        # and (in initialize_db) only ever wipe a dir that is empty or verifiably
        # a MySQL datadir.
        workspace = os.path.abspath(os.path.expanduser(workspace))
        if workspace == '/' or workspace == os.path.abspath(os.path.expanduser('~')):
            raise RuntimeError(
                f"VILLAGESQL_DB_WORKSPACE points at an unsafe path ({workspace!r}); "
                "use a dedicated disposable scratch directory (its data/ subdir is "
                "wiped on each run).")
        self._data_dir = os.path.join(workspace, 'data')
        self._log_file = os.path.join(workspace, 'villagesql.err')
        os.makedirs(self._data_dir, exist_ok=True)

        # Socket under /tmp to keep the path under the 107-char limit.
        dset = inspect.stack()[3].frame.f_locals['dataset_name'] + '-'
        self._socket_file = tempfile.mktemp(prefix=dset, suffix='.sock', dir='/tmp')

        print("\nSetup paths:")
        print(f"VILLAGESQL_ROOT_DIR: {root_dir}")
        print(f"DATA_DIR: {self._data_dir}")
        print(f"LOG_FILE: {self._log_file}")
        print(f"SOCKET_FILE: {self._socket_file}\n")

        # Command to initialize a fresh datadir (insecure = no root password).
        self._init_cmd = [
            self._mysqld,
            "--no-defaults",
            "--initialize-insecure",
            f"--basedir={self._basedir}",
            f"--datadir={self._data_dir}",
        ]

        # Command to start the server.
        self._start_cmd = [
            self._mysqld,
            "--no-defaults",
            f"--basedir={self._basedir}",
            f"--datadir={self._data_dir}",
            f"--socket={self._socket_file}",
            f"--log-error={self._log_file}",
            "--skip-networking",
            "--secure-file-priv=",
            # villagesql's HNSW graph is served from the InnoDB buffer pool (it
            # has no separate graph cache), so size the pool to match the total
            # memory MariaDB's adapter gets: 16G buffer pool + 10G MHNSW cache =
            # 26G. Keeps the two engines on an equal memory budget.
            "--loose-innodb-buffer-pool-size=26G",
            # Large redo log so the 60k-vector build's write burst does not
            # trigger checkpoint stalls (default is ~100MB). Matched with the
            # MariaDB adapter's innodb_log_file_size=2G.
            "--loose-innodb-redo-log-capacity=2G",
        ]
        user_option = VillageSQL.get_user_option()
        if user_option is not None:
            self._start_cmd += user_option
        self._mysqld_proc = None

    @staticmethod
    def looks_like_mysql_datadir(path):
        # A MySQL/VillageSQL datadir created by --initialize always contains the
        # InnoDB system tablespace and system schema. Key off those so we only
        # ever wipe a directory that is genuinely a datadir (ours or a prior
        # run's) — never arbitrary files a mistyped env var points at.
        entries = set(os.listdir(path))
        has_innodb = 'ibdata1' in entries or 'mysql.ibd' in entries
        has_system_schema = os.path.isdir(os.path.join(path, 'mysql'))
        return has_innodb and has_system_schema

    def initialize_db(self):
        try:
            if self._do_init:
                # --initialize-insecure refuses a non-empty datadir. Each
                # M/ef_construction combo instantiates a fresh VillageSQL against
                # the same workspace, so wipe the datadir first — otherwise the
                # second combo aborts with "data directory has files in it".
                print("\nInitializing VillageSQL datadir...")
                if os.path.isdir(self._data_dir) and os.listdir(self._data_dir):
                    # Non-empty: only wipe if it is unmistakably a MySQL datadir.
                    # Otherwise the env var is pointing somewhere it shouldn't and
                    # we must not destroy whatever is there.
                    if not VillageSQL.looks_like_mysql_datadir(self._data_dir):
                        raise RuntimeError(
                            f"Refusing to wipe {self._data_dir!r}: it is not empty and "
                            "does not look like a MySQL datadir (no ibdata1/mysql.ibd + "
                            "mysql/). Point VILLAGESQL_DB_WORKSPACE at a disposable "
                            "scratch directory.")
                    shutil.rmtree(self._data_dir)
                os.makedirs(self._data_dir, exist_ok=True)
                print(self._init_cmd)
                init_proc = subprocess.Popen(self._init_cmd, stdout=sys.stdout, stderr=sys.stderr)
                init_proc.wait()
        except Exception as e:
            print("ERROR: Failed to initialize VillageSQL datadir:", e)
            raise

    @staticmethod
    def get_user_option():
        try:
            return ["--user=root"] if getpass.getuser() == "root" else None
        except Exception:
            print("Could not get current user, could be docker user mapping. Ignore.")
            return None

    @staticmethod
    def can_run_perf():
        try:
            subprocess.run(["perf", "record", "echo", "testing perf"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except FileNotFoundError:
            print("Warning: perf command not found. Skipping.")
        except Exception as e:
            print(f"Warning: perf does not have permission to run. Skipping. Error: {e}")
        return False

    @staticmethod
    def can_run_flamegraph():
        if not VillageSQL.can_run_perf():
            return False
        if not shutil.which("stackcollapse-perf.pl"):
            print("Warning: Command 'stackcollapse-perf.pl' missing. Skipping.")
            return False
        if not shutil.which("flamegraph.pl"):
            print("Warning: Command 'flamegraph.pl' missing. Skipping.")
            return False
        return True

    def perf_start(self, name):
        if not self._perf_record and not self._perf_stat:
            return
        self.perf_stop()
        if self._perf_record:
            record_name = f"perf.data.{name}.{self._test_time}"
            self._perf_proc = subprocess.Popen([
                "perf", "record", "-p", f"{self._mysqld_proc.pid}", "-g",
                "--freq=100", "--output=results/" + record_name,
            ], stdout=sys.stdout, stderr=sys.stderr)
            self._perf_records.append(record_name)
        elif self._perf_stat:
            stat_name = f"perf.stat.{name}.{self._test_time}"
            self._perf_proc = subprocess.Popen([
                "perf", "stat", "-x,", f"--output=results/{stat_name}",
                "-p", f"{self._mysqld_proc.pid}"
            ], stdout=sys.stdout, stderr=sys.stderr)
            self._perf_stats.append(stat_name)

    def perf_stop(self):
        if (self._perf_record or self._perf_stat) and self._perf_proc is not None:
            self._perf_proc.send_signal(signal.SIGINT)
            try:
                self._perf_proc.wait(10)
                print("\nPerf process terminated.")
            except subprocess.TimeoutExpired:
                print("\nError: Perf process did not terminate within the timeout period.")
            self._perf_proc = None

    def perf_analysis(self):
        if self._perf_record:
            for record in self._perf_records:
                try:
                    flamegraph_cmd = f"perf script -i results/{record} | stackcollapse-perf.pl | flamegraph.pl > results/{record}.svg"
                    subprocess.run(flamegraph_cmd, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
                except subprocess.CalledProcessError as e:
                    print(f"Error: Failed to generate flame graph. Command '{e.cmd}' returned non-zero exit status {e.returncode}.")
        if self._perf_stat:
            for stat_file in self._perf_stats:
                try:
                    with open(f"results/{stat_file}", 'r') as file:
                        values = [int(line.split(',')[0]) for line in file if 'cpu_core/cycles/' in line or 'cpu_atom/cycles/' in line]
                        print(f"CPU cycles in {stat_file}: {sum(values):,.0f}" if values else "Error: No CPU cycle data found.")
                except (FileNotFoundError, IOError):
                    print("Error reading the perf stat file.")

    def start_db(self):
        try:
            print("\nStarting VillageSQL server...")
            print(self._start_cmd)
            self._mysqld_proc = subprocess.Popen(self._start_cmd, stdout=sys.stdout, stderr=sys.stderr)
        except Exception as e:
            print("ERROR: Failed to start VillageSQL server:", e)
            raise

        # Wait for the socket to accept connections (<=30s).
        start_time = time.time()
        while True:
            if time.time() - start_time > 30:
                raise TimeoutError("Timeout waiting for VillageSQL server to start")
            if os.path.exists(self._socket_file):
                try:
                    pymysql.connect(unix_socket=self._socket_file, user="root").close()
                    print("\nVillageSQL server started!")
                    break
                except pymysql.Error:
                    pass
            time.sleep(1)

        # Apply the gates the custom KNN path needs, then install the extension.
        # A fresh datadir means the extension is always (re)installed here.
        gate = pymysql.connect(unix_socket=self._socket_file, user="root")
        gcur = gate.cursor()
        gcur.execute("SET PERSIST vsql_allow_preview_extensions = ON")
        gcur.execute("SET GLOBAL optimizer_switch = 'hypergraph_optimizer=on'")
        try:
            gcur.execute("INSTALL EXTENSION vsql_vector")
        except pymysql.Error as e:
            print(f"(note: INSTALL EXTENSION vsql_vector: {e} — may already be installed)")
        gate.commit()
        gate.close()

    def fit(self, X):
        # Enable the custom KNN path on the harness's own connection too.
        self._cur.execute("SET optimizer_switch = 'hypergraph_optimizer=on'")

        print("\nPreparing database and table...")
        self._cur.execute("DROP DATABASE IF EXISTS ann")
        self._cur.execute("CREATE DATABASE ann")
        self._cur.execute("USE ann")
        self._cur.execute(f"""
          CREATE TABLE t1 (
            id INT PRIMARY KEY,
            v SVECTOR({len(X[0])}) NOT NULL
          ) ENGINE=InnoDB
        """)

        # Build the HNSW index BEFORE inserting so the graph is built
        # incrementally as rows arrive (vsql_vector's natural path). The insert
        # phase then IS the build phase, which is what we time below.
        print("\nCreating index...")
        self._cur.execute(
            f"CREATE INDEX idx_v ON t1 (v {self._modifier}) USING EXTENDED(hnsw) "
            f"WITH (M = {self._m}, ef_construction = {self._ef_construction})")

        print("\nInserting data (building index incrementally)...")
        start_time = time.time()
        self.perf_start("inserting")
        if self._batch:
            XX = []
            ncpu = os.cpu_count()
            for i in range(ncpu):
                n = int(len(X) / ncpu * i)
                XX.append((self._socket_file, n, X[n:int(len(X) / ncpu * (i + 1))]))
            pool = Pool()
            pool.map(many_inserts, XX)
        else:
            rps, rows, last, total = 1000, 0, time.time(), 1
            for i, embedding in enumerate(X):
                self._cur.execute("INSERT INTO t1 (id, v) VALUES (%s, %s)",
                                  (i, vector_to_literal(embedding)))
                if i - rows > rps:
                    now = time.time()
                    rps = ((i - rows) / (now - last) + 19 * rps) / 20
                    eta = (len(X) - i) / rps
                    total = now - start_time + eta
                    last, rows = now, i
                    print(f"{i:6_} of {len(X):_}, {rps:4.2f} stmt/sec ETA {eta:.0f} of {total:.0f} sec")
            self._cur.execute("commit")
        self.perf_stop()
        print(f"\nInsert+build time for {X.size:_} floats: {time.time() - start_time:7.2f}s")

        # Approximate on-disk index size from the table + index tablespaces.
        stem = f"{self._data_dir}/ann/"
        for f in glob.glob(stem + '*.ibd'):
            self._size += os.stat(f).st_size

        self.perf_start("searching")

    def set_query_arguments(self, ef_search):
        self._ef_search = ef_search
        self._cur.execute("SET vsql_vector.ef_search = %s" % ef_search)

    def query(self, v, n):
        self._cur.execute(
            f"SELECT id FROM t1 ORDER BY {self._dist_fn}(v, '{vector_to_literal(v)}') {self._order} LIMIT {n}")
        return [row[0] for row in self._cur.fetchall()]

    def get_memory_usage(self):
        return self._size / 1024  # kB

    def batch_query(self, X, n):
        XX = []
        ncpu = os.cpu_count()
        for i in range(ncpu):
            chunk = X[int(len(X) / ncpu * i):int(len(X) / ncpu * (i + 1))]
            XX.append((self._socket_file, self._ef_search, self._dist_fn, self._order, n, chunk))
        pool = Pool()
        self._res = pool.map(many_queries, XX)

    def get_batch_results(self):
        return chain(*self._res)

    def __str__(self):
        return f"VillageSQL(m={self._m:2d}, ef_construction={self._ef_construction}, ef_search={self._ef_search})"

    def done(self):
        self._cur.execute("shutdown")
        self._mysqld_proc.wait(300)
        self.perf_stop()
        self.perf_analysis()
