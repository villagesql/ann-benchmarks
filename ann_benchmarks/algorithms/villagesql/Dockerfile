FROM ann-benchmarks

# VillageSQL Server is a large fork of MySQL 8.4 and the vsql_vector extension
# is a separate build; there is no upstream apt package or short clone-and-build
# recipe as there is for MariaDB. Building both inside this image is possible but
# slow and version-specific, so the recommended path is to run this adapter
# LOCALLY against a prebuilt VillageSQL tree (see algorithms/villagesql/README):
#
#   VILLAGESQL_ROOT_DIR=/path/to/villagesql/build \
#   VILLAGESQL_DB_WORKSPACE=/tmp/vsql-ann \
#   python run.py --local --algorithm villagesql --dataset <dataset>
#
# To containerise instead, bind-mount a prebuilt tree into the image and set the
# two env vars below to point at it, then skip datadir init on first start:
#
#   ENV VILLAGESQL_ROOT_DIR /opt/villagesql/build
#   ENV VILLAGESQL_DB_WORKSPACE /home/villagesql
#   ENV DO_INIT_VILLAGESQL yes

# Python MySQL-protocol driver (VillageSQL speaks the MySQL protocol).
RUN pip3 install pymysql

WORKDIR /home/app
