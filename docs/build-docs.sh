#!/usr/bin/env bash

# Some general conf
NAME="zonopy-sphinx"
IMAGE="roahmlab/zonopy-sphinx:latest"

# Identify the script directory
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )"

## First build the docs container
tar -cf - -C "$SCRIPT_DIR/docs" Dockerfile requirements.txt | docker build -t $IMAGE -

## Configuration for script vars
MOUNT_DIR="$SCRIPT_DIR"
STARTING_DIR="$SCRIPT_DIR"
USE_UNIQUE=true
ADD_UNAME=true
if $USE_UNIQUE;then
    NAME+="-$(uuidgen)"
fi
if $ADD_UNAME;then
    NAME="$(id -un)-$NAME"
fi

## Setup uid requirements and workdir for temporaries
if [ -z "$HOME" ];then
    HOME=/tmp
fi
if [ -z "$ID" ];then
    ID=$(id -u)
fi
WORKDIR="$HOME/.docker"
mkdir -p "$WORKDIR"
DOCKER_HOME="$WORKDIR/$NAME"
mkdir -p "$DOCKER_HOME"

## Build out the Docker options
DOCKER_OPTIONS=""
DOCKER_OPTIONS+="-it "
DOCKER_OPTIONS+="--rm "

## USER ACCOUNT STUFF
DOCKER_OPTIONS+="--user $(id -u):$(id -g) "
DOCKER_OPTIONS+="$(id -G | sed -E "s/([[:blank:]]|^)([[:alnum:]_]+)/--group-add \2 /g") "
DOCKER_OPTIONS+="-e HOME=$HOME "

## PROJECT
DOCKER_OPTIONS+="-v $MOUNT_DIR:/zonopy "
DOCKER_OPTIONS+="-v $DOCKER_HOME:$HOME "
DOCKER_OPTIONS+="--name $NAME "
DOCKER_OPTIONS+="--workdir=/zonopy/docs "
DOCKER_OPTIONS+="--entrypoint make "
DOCKER_OPTIONS+="--net=host "

## CLEANUP
function cleanup {      
  rm -rf "$DOCKER_HOME"
}
trap cleanup EXIT

## RUN
docker run $DOCKER_OPTIONS $IMAGE $1
