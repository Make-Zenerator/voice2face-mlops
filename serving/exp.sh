docker compose down
docker rmi $(docker images)
docker compose -f docker-compose.yaml up -d