COMPOSE = docker compose -f docker/docker-compose.yml
# Load .env for Make variables (e.g. neo4j-schema). Keep lines in shell/env form: KEY=value.
-include .env
export

NEO4J_USER ?= neo4j
NEO4J_PASSWORD ?= neo4jpassword

.PHONY: compose-reset up up-all down reset logs ps

compose-reset:
	$(COMPOSE) down --remove-orphans
	@echo "Stale stack cleared. Retry: make up"

# Core services (no Elasticsearch — avoids vm.max_map_count bootstrap failures on many hosts).
up:
	$(COMPOSE) up -d
	@echo "Services starting... check status with: make ps"
	@echo "Note: Elasticsearch is optional. For full stack incl. ES: make up-all"

# Full stack including Elasticsearch (host may need: sudo sysctl -w vm.max_map_count=262144)
up-all:
	$(COMPOSE) --profile es up -d
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
	cat src/graph/schema.cypher | $(COMPOSE) exec -T neo4j \
		cypher-shell -u "$(NEO4J_USER)" -p "$(NEO4J_PASSWORD)"

# ── Tests ──────────────────────────────────────────────────

test:
	python -m pytest tests/ -v

.PHONY: compose-reset ingest-batch ingest-stream build-graph run-er neo4j-schema test
