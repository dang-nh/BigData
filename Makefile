COMPOSE = docker compose -f docker/docker-compose.yml

.PHONY: up down reset logs ps

up:
	$(COMPOSE) up -d
	@echo "Services starting... check status with: make ps"

down:
	$(COMPOSE) down

reset:
	$(COMPOSE) down -v
	@echo "All volumes removed."

logs:
	$(COMPOSE) logs -f

ps:
	$(COMPOSE) ps

# ── Data pipeline shortcuts ────────────────────────────────

ingest-batch:
	docker exec spark-master spark-submit \
		--master spark://spark-master:7077 \
		--packages org.mongodb.spark:mongo-spark-connector_2.12:10.3.0 \
		/app/src/ingest/batch_loader.py

ingest-stream:
	docker exec spark-master spark-submit \
		--master spark://spark-master:7077 \
		--packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,org.mongodb.spark:mongo-spark-connector_2.12:10.3.0 \
		/app/src/ingest/kafka_consumer.py

build-graph:
	docker exec spark-master spark-submit \
		--master spark://spark-master:7077 \
		--packages org.neo4j:neo4j-connector-apache-spark_2.12:5.3.0_for_spark_3 \
		/app/src/graph/graph_builder.py

run-er:
	docker exec spark-master spark-submit \
		--master spark://spark-master:7077 \
		/app/src/er/classifier.py

# ── Neo4j schema setup ─────────────────────────────────────

neo4j-schema:
	docker exec neo4j cypher-shell -u neo4j -p neo4jpassword -f /var/lib/neo4j/import/schema.cypher

# ── Tests ──────────────────────────────────────────────────

test:
	python -m pytest tests/ -v

.PHONY: ingest-batch ingest-stream build-graph run-er neo4j-schema test
