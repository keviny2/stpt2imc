#!/bin/bash
pwd
for index in {01..18}
do
	echo "$index"
	mv "SECTION_$index/TIF/"* "SECTION_$index"
	rm -r "SECTION_$index/ZARR"
	rm -r "SECTION_$index/TIF"
done
