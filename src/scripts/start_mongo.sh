mongod --dbpath data/shard1 --port 27000 --fork --syslog
mongod --dbpath data/shard2 --port 27001 --fork --syslog
mongod --configsvr --dbpath data/config --port 27002 --fork --syslog
mongos --configdb localhost:27002 --port 27100 --fork --syslog
