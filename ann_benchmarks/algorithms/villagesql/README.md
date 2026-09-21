# VillageSQL (vsql_vector) adapter

Runs [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) against the
`vsql_vector` extension (SVECTOR type + HNSW custom index) on VillageSQL Server.
It is adapted from the `mariadb` adapter so the two can be compared under the
same harness, datasets, and `plot.py` — the same setup the
[MariaDB Vector blog post](https://mariadb.com/resources/blog/how-fast-is-mariadb-vector/)
used.

## What differs from the MariaDB adapter

| | MariaDB | VillageSQL |
|---|---|---|
| Driver | `mariadb` connector | `pymysql` (MySQL protocol) |
| Column | `VECTOR(N)` | `SVECTOR(N)` on `ENGINE=InnoDB` |
| Value format | raw float32 bytes | text literal `'[f0,f1,...]'` |
| Index | inline `VECTOR INDEX (v)` in `CREATE TABLE` | separate `CREATE INDEX idx_v ON t (v hnsw_l2) USING EXTENDED(hnsw) WITH (M=…, ef_construction=…)` |
| Query breadth | `SET mhnsw_ef_search = N` | `SET vsql_vector.ef_search = N` |
| ef_construction | fixed (10) | tunable |
| Extra gates | none | `SET PERSIST vsql_allow_preview_extensions = ON`, `SET GLOBAL optimizer_switch='hypergraph_optimizer=on'`, `INSTALL EXTENSION vsql_vector` |

The HNSW index is created **before** inserting so the graph builds
incrementally as rows arrive (vsql_vector's natural path); the insert phase is
therefore the build phase, which is what the harness times.

## Running locally

This adapter starts its own `mysqld` on a scratch datadir + unix socket, so run
it with `--local` (no Docker). You need:

1. A built VillageSQL tree — the directory containing
   `runtime_output_directory/mysqld` and `veb_output_directory/`.
2. The `vsql_vector` VEB available. `INSTALL EXTENSION vsql_vector` resolves it
   by name from the server tree's `veb_output_directory/`, so either it is
   already there (via the extension's `make install`) **or** you point
   `VSQL_VECTOR_VEB` at a freshly built `vsql_vector.veb` and the adapter copies
   it in on startup. Prefer the latter — otherwise you may silently benchmark a
   stale VEB left in the server tree.
3. `pymysql` in the host environment (`pip install -r requirements-host.txt`).

```bash
export VILLAGESQL_ROOT_DIR=/path/to/villagesql/build
export VILLAGESQL_DB_WORKSPACE=/tmp/vsql-ann        # scratch datadir + error log
export VSQL_VECTOR_VEB=/path/to/vsql-vector/build/vsql_vector.veb  # exact VEB to test

python run.py --local --algorithm villagesql \
    --dataset fashion-mnist-784-euclidean

# batch (48-way concurrent) mode, as in the MariaDB blog:
python run.py --local --batch --algorithm villagesql \
    --dataset fashion-mnist-784-euclidean

python plot.py --dataset fashion-mnist-784-euclidean
```

Datasets from the blog: `mnist-784-euclidean`, `fashion-mnist-784-euclidean`,
`sift-128-euclidean`, `gist-960-euclidean`. On first use `run.py` downloads the
HDF5 into `data/`. If you already have the standard ann-benchmarks HDF5 (same
schema: `train`/`test`/`neighbors`/`distances` + `distance` attr), just drop or
symlink it in to skip the download:

```bash
ln -s /path/to/fashion-mnist-784-euclidean.hdf5 data/fashion-mnist-784-euclidean.hdf5
```

To constrain a run while iterating, add `--max-n-algorithms 1 --runs 1` (one
M×ef_construction combo, one pass).

### Environment variables

| Variable | Meaning |
|---|---|
| `VILLAGESQL_ROOT_DIR` | VillageSQL build dir (has `runtime_output_directory/mysqld`). Required. |
| `VILLAGESQL_DB_WORKSPACE` | Scratch dir for the datadir and error log. Required. |
| `VSQL_VECTOR_VEB` | Path to the `vsql_vector.veb` to benchmark; copied into the server's `veb_output_directory` on startup. Optional — if unset, the VEB already in the server tree is used. |
| `DO_INIT_VILLAGESQL` | `yes` (default) initializes a fresh datadir with `--initialize-insecure`. Set `no` if the datadir is pre-initialized. |

**Datadir wipe safety.** `--initialize-insecure` refuses a non-empty datadir, and
each M/ef_construction combo re-initializes, so the adapter wipes
`VILLAGESQL_DB_WORKSPACE/data` before each init. This is guarded so a mistyped
env var can't destroy unrelated data: it refuses a workspace of `/` or your home
directory, and it only ever `rmtree`s the `data/` subdir when that dir is empty
**or** verifiably a MySQL datadir (contains `ibdata1`/`mysql.ibd` and a `mysql/`
system schema). A non-empty dir that is not a datadir aborts the run instead of
being deleted. Always point `VILLAGESQL_DB_WORKSPACE` at a dedicated disposable
scratch directory.
| `PERF` / `FLAMEGRAPH` | `yes` to attach `perf stat` / generate a flame graph around insert/index/search phases (Linux only). |

## Notes

- Only `euclidean` and `angular` (→ cosine) metrics are wired up, matching the
  blog's datasets. Add rows to `_METRICS` in `module.py` for L1 / inner product.
- Index size (`get_memory_usage`) is approximated from the `.ibd` tablespaces
  under the `ann` schema.
