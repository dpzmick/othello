#!/bin/bash

cd /var/nfs/dpzmick
source env/bin/activate

cd /var/nfs/dpzmick/othello/
"$@"
