#!/bin/bash

for pdb in $(ls pdb/*.pdb)
do
    echo $pdb
    dssp $pdb > dssp/$(basename $pdb .pdb).dssp
done
