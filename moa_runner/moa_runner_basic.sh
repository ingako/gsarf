#!/bin/bash

BASEDIR=`dirname $0`/..
BASEDIR=`(cd "$BASEDIR"; pwd)`
MEMORY=512m

java -Xmx$MEMORY -cp "$BASEDIR/lib/moa-2018.6.0:$BASEDIR/lib/*" -javaagent:$BASEDIR/lib/sizeofag-1.0.4.jar moa.DoTask \
"EvaluatePrequential -l (meta.AdaptiveRandomForest -l (ARFHoeffdingTree -k $2 -e 2000000 -g 1000 -c 0.05 -t 0.0 -b -p -l MC) -s 100 -o (Specified m (integer value)) -m $2 -a 1.0 -x (ADWINChangeDetector -a 0.001) -p (ADWINChangeDetector -a 0.001) -w -q) -s (clustering.SimpleCSVStream -f $1 -c -t 0.0) -i 1000000 -f 1000"
