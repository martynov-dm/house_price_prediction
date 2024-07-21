run-api:
	python -m app.main
run-frontend:
	cd frontend && npm run dev
docker-run-api:
	docker build -t mfdp-api . && docker run --name mfdp-api -p 80:80 -it --rm mfdp-ap
docker-stop:
	docker stop mfdp-api
	docker rm mfdp-api
up:
	docker-compose up --build --force-recreate --no-deps
