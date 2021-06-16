# news-analyzer
MongoDB course work

### Used Kaggle dataset - [[link]](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
(I've added column ```target``` to define is news true(1) or fake(0), and merged two files. Total file size is to big, that's why it's not here)

### Mongo Components

* Config Server (3 member replica set): `configsvr01`,`configsvr02`,`configsvr03`

* 3 Shards (each a 3 member `PSA` replica set):
	* `shard01-a`,`shard01-b` and 1 arbiter `shard01-x`
	* `shard02-a`,`shard02-b` and 1 arbiter `shard02-x`
	* `shard03-a`,`shard03-b` and 1 arbiter `shard03-x`

* 2 Routers (mongos): `router01`, `router02`

<img src="https://raw.githubusercontent.com/minhhungit/mongodb-cluster-docker-compose/master/images/sharding-and-replica-sets.png" style="width: 100%;" />
