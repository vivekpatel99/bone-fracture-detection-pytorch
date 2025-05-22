.PHONY: build run all clean

IMAGE_NAME := cancer-cls
HOST_PORT := 8000
CONTAINER_PORT := 8000

build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run \
		-it \
		--env-file .env \
		--rm \
		--ulimit memlock=-1:-1 \
		--ulimit nofile=65536:65536 \
		--ipc=host \
		-p $(HOST_PORT):$(CONTAINER_PORT) \
		$(IMAGE_NAME)



all: build run

clean:
	docker rmi $(IMAGE_NAME) || true
