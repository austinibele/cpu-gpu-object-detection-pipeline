#!/bin/bash

docker-compose run --rm cpu_train python -m src.training.training_session.training_session "$@"
