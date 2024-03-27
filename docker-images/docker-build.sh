
#echo $1 | cut -d "_" -f 2
IN=$1
arrIN=(${IN//_/ })
NAME=${arrIN[1]}

if [ -z "$NAME" ]; then
    echo "provide correctly named Dockerfile; e.g. ./docker-build.sh Dockerfile_cp311-st26";
else
    docker build -t $NAME -f Dockerfile_$NAME .;
fi
