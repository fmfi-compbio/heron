# Heron - high accuracy GPU nanopore basecaller

## Instalation

* Install trans-decoder (`https://github.com/fmfi-compbio/transducer_decoder`)
* Install requirements `pip install -r requirements.txt`
* Download `http://compbio.fmph.uniba.sk/heron/network.pth` into weights directory.

## Running 

`python3 basecall.py --directory <directory_with_reads> --output some.fasta`
