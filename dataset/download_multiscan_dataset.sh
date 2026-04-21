#!/bin/bash

OBJECT_INSTANCE_SEGMENTATION=false
ARTICULATED_OBJECTS=false
PART_INSTANCE_SEGMENTATION=false
SHAPE2MOTION=false
OPDPN=false

function download {
    if [ $OBJECT_INSTANCE_SEGMENTATION = true ]; then
        echo "Downloading object instance segmentation dataset to ${OUTPUT_DIR}..."
        wget https://aspis.cmpt.sfu.ca/projects/multiscan/benchmark_dataset/object_instance_segmentation.zip -nc -P $OUTPUT_DIR
    fi

    if [ $ARTICULATED_OBJECTS = true ]; then
        echo "Downloading articulated objects dataset to ${OUTPUT_DIR}..."
        wget https://aspis.cmpt.sfu.ca/projects/multiscan/benchmark_dataset/articulated_dataset.zip -nc -P $OUTPUT_DIR
    fi

    if [ $PART_INSTANCE_SEGMENTATION = true ]; then
        echo "Downloading part instance segmentation dataset to ${OUTPUT_DIR}..."
        wget https://aspis.cmpt.sfu.ca/projects/multiscan/benchmark_dataset/part_instance_segmentation.zip -nc -P $OUTPUT_DIR
    fi

    if [ $SHAPE2MOTION = true ]; then
        echo "Downloading shape2motion input dataset to ${OUTPUT_DIR}..."
        wget https://aspis.cmpt.sfu.ca/projects/multiscan/benchmark_dataset/shape2motion.zip -nc -P $OUTPUT_DIR
    fi

    if [ $OPDPN = true ]; then
        echo "Downloading opdpn input dataset to ${OUTPUT_DIR}..."
        wget https://aspis.cmpt.sfu.ca/projects/multiscan/benchmark_dataset/opdpn.zip -nc -P $OUTPUT_DIR
    fi

}

while true; do
    case "$1" in
        -h)
            echo "Usage: $0 [-o|-a|-p|-s|-n] <output_dir>"
            echo "download multiscan benchmark dataset"
            echo "[-h] prints this help message and quits"
            echo "[-o|--object_instance_segmentation] object instance segmentation dataset"
            echo "[-a|--articulated_objects] articulated objects dataset"
            echo "[-p|--part_instance_segmentation] part instance segmentation dataset"
            echo "[-s|--shape2motion] shape2motion input dataset"
            echo "[-n|--opdpn] opdpn input dataset"
            exit 1
            ;;
        -o|--object_instance_segmentation)
            OBJECT_INSTANCE_SEGMENTATION=true
            shift
            ;;
        -a|--articulated_objects)
            ARTICULATED_OBJECTS=true
            shift
            ;;
        -p|--part_instance_segmentation)
            PART_INSTANCE_SEGMENTATION=true
            shift
            ;;
        -s|--shape2motion)
            SHAPE2MOTION=true
            shift
            ;;
        -n|--opdpn)
            OPDPN=true
            shift
            ;;
        *)
            OUTPUT_DIR="$1"
            shift
            break
    esac
done

if [ -z $OUTPUT_DIR ]
then
   echo "Please specify output directory"
   exit 1
fi

download