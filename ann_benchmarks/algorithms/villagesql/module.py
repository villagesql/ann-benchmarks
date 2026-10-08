import getpass
import glob
import os
import shutil
import signal
import struct
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
#   * One server-side gate is required: preview extensions must be allowed, to
#     INSTALL the preview extension. The optimizer stays CLASSIC.

# Metric -> (index modifier used at build time, distance function used at query
# time, sort order). The modifier and the distance function must name the same
# metric or the index is not used.
_METRICS = {
    "euclidean": {"modifier": "hnsw_l2", "dist_fn": "L2_DISTANCE", "order": "ASC"},
    "cosine": {"modifier": "hnsw_cosine", "dist_fn": "COSINE_DISTANCE", "order": "ASC"},
}


# How a vector is handed to the server.
#
#   text    the decimal literal '[f0,f1,...]'.
#   binary  the SVECTOR binary form: a 2-byte big-endian element count with its
#           high bit set, then the little-endian float32 elements. Much cheaper
#           to build client-side than a decimal literal, whose cost can
#           dominate a query.
#
# Set VILLAGESQL_VECTOR_ENCODING to pick one; text is the default.
VECTOR_ENCODING = os.environ.get('VILLAGESQL_VECTOR_ENCODING', 'text')

# Which client library to talk to the server with. Set VILLAGESQL_DRIVER:
#
#   pymysql    PyMySQL, the default. Text protocol only, so no prepared
#              statements.
#   connector  mysql-connector-python. Can prepare statements.
#   mariadb    MariaDB Connector/Python, which is what the mariadb adapter in
#              this repo uses, so it is the like-for-like client for comparing
#              against MariaDB's own numbers. Prepares by default (its
#              execute() is a prepare-and-execute) and uses ? placeholders
#              rather than %s. Needs libmariadb-dev to install.
#
# This is a separate knob from VILLAGESQL_USE_PREPARED on purpose. PyMySQL has
# no prepared-statement path, so turning prepared statements on also changes
# the client library, and a run that varied only USE_PREPARED would measure the
# two together. Holding the driver fixed and varying USE_PREPARED measures
# prepared statements alone; holding USE_PREPARED at no and varying the driver
# measures the library alone.
DRIVER = os.environ.get('VILLAGESQL_DRIVER', 'pymysql')

# Whether the per-vector statements are prepared once and re-executed, instead
# of being built as a fresh SQL string per row.
USE_PREPARED = os.environ.get('VILLAGESQL_USE_PREPARED', 'no') == 'yes'

_DRIVERS = ('pymysql', 'connector', 'mariadb')
_PREPARING_DRIVERS = ('connector', 'mariadb')

# The placeholder each driver's paramstyle wants. Only matters where a value is
# bound rather than interpolated, i.e. in prepared mode.
PLACEHOLDER = '?' if DRIVER == 'mariadb' else '%s'

if DRIVER not in _DRIVERS:
    raise RuntimeError(
        f"VILLAGESQL_DRIVER={DRIVER!r} is not one of {sorted(_DRIVERS)}")
if USE_PREPARED and DRIVER not in _PREPARING_DRIVERS:
    raise RuntimeError(
        f"VILLAGESQL_USE_PREPARED=yes needs one of "
        f"{sorted(_PREPARING_DRIVERS)}; {DRIVER!r} cannot prepare statements")


def _vector_to_text_literal(v):
    # Quoted so it drops into SQL exactly as the binary form does. Full
    # float32 precision.
    return "'[" + ",".join(repr(float(x)) for x in v) + "]'"


def _vector_to_binary_literal(v):
    # _binary X'..' rather than a %s parameter because PyMySQL escapes a bytes
    # parameter to a bare X'..', which is coerced via the connection charset
    # and truncates at the first NUL.
    a = numpy.asarray(v, '<f4')
    header = struct.pack('>H', a.size | 0x8000)
    return "_binary X'" + header.hex() + a.tobytes().hex() + "'"


# Prepared mode binds the vector as a parameter instead, so these return a
# value rather than a SQL fragment: no quoting, no _binary prefix, no hex.
def _vector_to_text_value(v):
    return "[" + ",".join(repr(float(x)) for x in v) + "]"


def _vector_to_binary_value(v):
    # Connector/Python declares a bytes parameter as FieldType.STRING, so the
    # value reaches the server with the connection charset rather than binary
    # and is handled by the type's string converter. The 0x8000 tag is what
    # lets that converter recognize it as the binary form, so it is still
    # required here. (Sending bare bytes to a from_binary hook instead needs
    # the COM_STMT_SEND_LONG_DATA route, which is a separate experiment.)
    a = numpy.asarray(v, '<f4')
    return struct.pack('>H', a.size | 0x8000) + a.tobytes()


_VECTOR_FRAGMENTS = {
    'text': _vector_to_text_literal,
    'binary': _vector_to_binary_literal,
}

_VECTOR_VALUES = {
    'text': _vector_to_text_value,
    'binary': _vector_to_binary_value,
}

if VECTOR_ENCODING not in _VECTOR_FRAGMENTS:
    raise RuntimeError(
        f"VILLAGESQL_VECTOR_ENCODING={VECTOR_ENCODING!r} is not one of "
        f"{sorted(_VECTOR_FRAGMENTS)}")


def vector_to_literal(v):
    """Render a vector as a SQL fragment in the configured encoding."""
    return _VECTOR_FRAGMENTS[VECTOR_ENCODING](v)


def vector_to_value(v):
    """Render a vector as a bound-parameter value in the configured encoding."""
    return _VECTOR_VALUES[VECTOR_ENCODING](v)


# The concurrent insert workers retry on a transient failure (lock wait,
# deadlock); each driver raises its own type for those.
if DRIVER == 'connector':
    import mysql.connector
    RETRY_ERRORS = (mysql.connector.errors.OperationalError,
                    mysql.connector.errors.DatabaseError)
elif DRIVER == 'mariadb':
    try:
        import mariadb
    except ImportError as e:
        raise RuntimeError(
            "VILLAGESQL_DRIVER=mariadb needs MariaDB Connector/Python, which "
            "builds against libmariadb: apt-get install libmariadb-dev, then "
            f"pip install mariadb ({e})") from e
    RETRY_ERRORS = (mariadb.OperationalError, mariadb.DatabaseError)
else:
    RETRY_ERRORS = (pymysql.OperationalError,)


def driver_info():
    """How the chosen driver is actually talking to the server.

    Reported so a run's numbers can be attributed: mysql-connector-python ships
    a C extension and a pure-Python implementation, and the pure one is
    substantially slower, so which is in use has to be recorded rather than
    assumed.
    """
    prepared = 'yes' if USE_PREPARED else 'no'
    if DRIVER == 'connector':
        impl = 'C extension' if mysql.connector.HAVE_CEXT else 'pure Python'
        return (f"mysql-connector-python {mysql.connector.__version__} "
                f"({impl}), prepared={prepared}")
    if DRIVER == 'mariadb':
        return (f"mariadb {mariadb.__version__} "
                f"(C ext over libmariadb), prepared={prepared}")
    return f"pymysql {pymysql.__version__}, prepared={prepared}"


def connect(socket_file):
    """Open a connection with the configured driver."""
    # autocommit on the prepared-capable drivers so the insert loop behaves as
    # it does under PyMySQL, which leaves autocommit on by default.
    if DRIVER == 'connector':
        return mysql.connector.connect(unix_socket=socket_file, user="root",
                                       autocommit=True)
    if DRIVER == 'mariadb':
        return mariadb.connect(unix_socket=socket_file, user="root",
                               autocommit=True)
    return pymysql.connect(unix_socket=socket_file, user="root")


def vector_cursor(conn):
    """A cursor for ONE per-vector statement, prepared when that mode is on.

    A prepared cursor prepares on its first execute and reuses the handle
    after, so it must be long-lived and must run only that one statement.
    Sending a second statement through it re-prepares, and with the C
    connector that leaves an unread result behind and breaks the connection,
    so the INSERT and the SELECT need one of these each. Anything else (USE,
    SET, DDL, commit) goes through plain_cursor().

    MariaDB Connector/Python has no prepared=True: its execute() is already a
    prepare-and-execute that caches the handle on the cursor, so an ordinary
    cursor is the prepared one and the same single-statement rule applies.
    """
    if not USE_PREPARED or DRIVER == 'mariadb':
        return conn.cursor()
    return conn.cursor(prepared=True)


def plain_cursor(conn):
    """A cursor for one-off statements, never prepared."""
    return conn.cursor()


def many_inserts(arg):
    socket_file, base, embeddings = arg
    conn = connect(socket_file)
    plain_cursor(conn).execute("USE ann")
    cur = vector_cursor(conn)
    lenX = len(embeddings)
    start_time = time.time()
    rps = 100
    for i, embedding in enumerate(embeddings):
        while True:
            try:
                if USE_PREPARED:
                    cur.execute(
                        f"INSERT INTO t1 (id, v) VALUES "
                        f"({PLACEHOLDER}, {PLACEHOLDER})",
                        (i + base, vector_to_value(embedding)))
                else:
                    cur.execute(
                        f"INSERT INTO t1 (id, v) VALUES "
                        f"({PLACEHOLDER}, {vector_to_literal(embedding)})",
                        (i + base,))
                break
            except RETRY_ERRORS:
                time.sleep(0.01 * (11 + base * 17 % 13))
        if base == 0 and (i + 1) % int(rps + 1) == 0:
            rps = i / (time.time() - start_time)
            print(f"{i:6d} of {lenX}, {rps:4.2f} stmt/sec, ETA {(lenX - i) / rps:.0f} sec")
    plain_cursor(conn).execute("commit")


def many_queries(arg):
    socket_file, ef_search, dist_fn, order, n, queries = arg
    conn = connect(socket_file)
    setup = plain_cursor(conn)
    setup.execute("USE ann")
    setup.execute("SET vsql_vector.ef_search = %s" % ef_search)
    cur = vector_cursor(conn)
    res = []
    for v in queries:
        if USE_PREPARED:
            cur.execute(
                f"SELECT id FROM t1 ORDER BY {dist_fn}(v, {PLACEHOLDER}) "
                f"{order} LIMIT {n}",
                (vector_to_value(v),))
        else:
            cur.execute(
                f"SELECT id FROM t1 ORDER BY {dist_fn}(v, {vector_to_literal(v)}) {order} LIMIT {n}")
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

        conn = connect(self._socket_file)
        # A prepared cursor holds exactly ONE statement: sending a second
        # statement through it re-prepares, and with the C connector that
        # leaves an unread result behind and breaks the connection. So the
        # INSERT and the SELECT each get their own, created lazily on first
        # use, and everything else (DDL, USE, SET, commit) stays on the plain
        # cursor.
        self._conn = conn
        self._cur = plain_cursor(conn)
        self._ins_cur = None
        self._qry_cur = None

    @staticmethod
    def buffer_pool_size():
        """Buffer pool to request, capped to something this machine can map."""
        explicit = os.environ.get('VILLAGESQL_BUFFER_POOL')
        if explicit:
            return explicit
        try:
            total = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
        except (ValueError, OSError, AttributeError):
            return "4G"
        # Half of RAM, clamped to [1G, 26G]: enough to hold these datasets
        # entirely while leaving room for the OS and the benchmark client.
        gb = max(1, min(26, int(total * 0.5) // (1024 ** 3)))
        return f"{gb}G"

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
        print(f"SOCKET_FILE: {self._socket_file}")
        print(f"DRIVER: {driver_info()}")
        print(f"VECTOR_ENCODING: {VECTOR_ENCODING}\n")

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
            # has no separate graph cache), so the pool must hold the whole
            # table + graph. The reference budget is what MariaDB's adapter
            # gets (16G pool + 10G MHNSW cache = 26G), but it must be capped to
            # the machine: requesting more than physical RAM makes InnoDB spin
            # at startup and never reach "ready for connections", and --loose-
            # hides it. Override with VILLAGESQL_BUFFER_POOL to force a size.
            f"--loose-innodb-buffer-pool-size={VillageSQL.buffer_pool_size()}",
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

        # Startup probing and the one-off gates below stay on PyMySQL whatever
        # VILLAGESQL_USE_PREPARED says: nothing here is per-vector, so the
        # driver makes no difference, and PyMySQL is always installed.
        #
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
        try:
            gcur.execute("INSTALL EXTENSION vsql_vector")
        except pymysql.Error as e:
            print(f"(note: INSTALL EXTENSION vsql_vector: {e} — may already be installed)")
        gate.commit()
        gate.close()

    def fit(self, X):
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
            if self._ins_cur is None:
                self._ins_cur = vector_cursor(self._conn)
            for i, embedding in enumerate(X):
                if USE_PREPARED:
                    self._ins_cur.execute(
                        f"INSERT INTO t1 (id, v) VALUES "
                        f"({PLACEHOLDER}, {PLACEHOLDER})",
                        (i, vector_to_value(embedding)))
                else:
                    self._ins_cur.execute(
                        f"INSERT INTO t1 (id, v) VALUES "
                        f"({PLACEHOLDER}, {vector_to_literal(embedding)})",
                        (i,))
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
        # n is interpolated rather than bound: it is part of the statement
        # shape (LIMIT), and ann-benchmarks holds it fixed for a run, so the
        # prepared handle is still reused across every query.
        if self._qry_cur is None:
            self._qry_cur = vector_cursor(self._conn)
        if USE_PREPARED:
            self._qry_cur.execute(
                f"SELECT id FROM t1 ORDER BY "
                f"{self._dist_fn}(v, {PLACEHOLDER}) {self._order} LIMIT {n}",
                (vector_to_value(v),))
        else:
            self._qry_cur.execute(
                f"SELECT id FROM t1 ORDER BY {self._dist_fn}(v, {vector_to_literal(v)}) {self._order} LIMIT {n}")
        return [row[0] for row in self._qry_cur.fetchall()]

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
        # The server drops the connection as it shuts down, which some drivers
        # surface as an error on the statement itself. The process wait below
        # is what actually confirms it stopped.
        try:
            self._cur.execute("shutdown")
        except Exception as e:
            print(f"(note: shutdown reported {type(e).__name__}: {e})")
        self._mysqld_proc.wait(300)
        self.perf_stop()
        self.perf_analysis()
